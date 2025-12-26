"""
PoseLift Dataset Loader for Shopformer_2.

Adapted for paper-aligned settings:
- seq_len=12 frames (paper spec)
- stride=6 (50% overlap)
- 18 keypoints (COCO-17 + synthetic neck keypoint for 144 embedding size)
- MPS-compatible DataLoader settings

Dataset Structure:
    DATA/PoseLift/
    ├── Pickle_files/
    │   ├── Train/
    │   │   └── <camera>_<video>.pkl
    │   ├── Test/
    │   │   └── <camera>_<video>.pkl
    │   └── GT/
    │       └── <camera>_<video>.npy

Each .pkl file contains:
    - Dictionary: {frame_num: {person_id: [bbox, keypoints_array]}}
    - keypoints_array shape: (17, 3) with [x, y, confidence]

Note: We add a synthetic "neck" keypoint (index 17) as the average of
left_shoulder (5) and right_shoulder (6) to match the paper's 18-keypoint
configuration and achieve embedding size 144 (8 * 18).
"""

import math
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path


# COCO keypoint indices for left-right swap
# Note: Neck (index 17) is central and doesn't need swapping
COCO_KEYPOINT_FLIP_PAIRS = [
    (1, 2),   # left_eye, right_eye
    (3, 4),   # left_ear, right_ear
    (5, 6),   # left_shoulder, right_shoulder
    (7, 8),   # left_elbow, right_elbow
    (9, 10),  # left_wrist, right_wrist
    (11, 12), # left_hip, right_hip
    (13, 14), # left_knee, right_knee
    (15, 16), # left_ankle, right_ankle
]

# Indices for computing synthetic neck keypoint
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
NECK_IDX = 17  # Added as 18th keypoint


def add_neck_keypoint(keypoints: np.ndarray) -> np.ndarray:
    """
    Add synthetic neck keypoint as average of left and right shoulders.

    The paper uses 18 keypoints (OpenPose COCO format) which includes a neck.
    Our dataset has 17 keypoints (COCO), so we synthesize the neck to match
    the paper's 144 embedding size (8 channels * 18 keypoints).

    Args:
        keypoints: Shape (17, C) where C is 2 or 3 (x, y, [confidence])

    Returns:
        Extended keypoints with neck: Shape (18, C)
    """
    if keypoints.shape[0] < 17:
        # Pad to 17 first if needed
        pad_size = 17 - keypoints.shape[0]
        keypoints = np.vstack([keypoints, np.zeros((pad_size, keypoints.shape[1]))])

    left_shoulder = keypoints[LEFT_SHOULDER_IDX]
    right_shoulder = keypoints[RIGHT_SHOULDER_IDX]

    # Compute neck as midpoint of shoulders
    neck = (left_shoulder + right_shoulder) / 2.0

    # Handle case where shoulders are missing (zeros)
    if np.allclose(left_shoulder[:2], 0) and np.allclose(right_shoulder[:2], 0):
        neck = np.zeros_like(left_shoulder)
    elif np.allclose(left_shoulder[:2], 0):
        neck = right_shoulder.copy()
    elif np.allclose(right_shoulder[:2], 0):
        neck = left_shoulder.copy()

    # Append neck as 18th keypoint
    return np.vstack([keypoints[:17], neck.reshape(1, -1)])


def get_affine_transform_matrix(
    sx: float = 1.0,
    sy: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    rot: float = 0.0,
    shearx: float = 0.0,
    sheary: float = 0.0,
    flip: bool = False
) -> np.ndarray:
    """
    Generate 3x3 affine transformation matrix.

    Args:
        sx, sy: Scale factors
        tx, ty: Translation
        rot: Rotation in degrees
        shearx, sheary: Shear factors
        flip: Horizontal flip

    Returns:
        3x3 transformation matrix
    """
    cos_r = math.cos(math.radians(rot))
    sin_r = math.sin(math.radians(rot))
    flip_val = -1.0 if flip else 1.0

    mat = np.array([
        [sx * flip_val * cos_r - sheary * sy * sin_r,
         shearx * sx * flip_val * cos_r - sy * sin_r,
         tx * cos_r - ty * sin_r],
        [sx * flip_val * sin_r + sheary * sy * cos_r,
         shearx * sx * flip_val * sin_r + sy * cos_r,
         tx * sin_r + ty * cos_r],
        [0, 0, 1]
    ], dtype=np.float32)

    return mat


def apply_affine_transform(pose_seq: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Apply affine transformation to pose sequence.

    Args:
        pose_seq: Shape (T, V, C) where C >= 2 (x, y, ...)
        transform_matrix: 3x3 affine transform matrix

    Returns:
        Transformed pose sequence
    """
    T, V, C = pose_seq.shape
    result = pose_seq.copy()

    coords = pose_seq[:, :, :2]
    ones = np.ones((T, V, 1))
    coords_h = np.concatenate([coords, ones], axis=-1)

    transformed = np.einsum('tvc,cd->tvd', coords_h, transform_matrix[:2, :].T)
    result[:, :, :2] = transformed

    return result


def flip_keypoints(pose_seq: np.ndarray, num_keypoints: int = 17) -> np.ndarray:
    """Swap left and right keypoints after horizontal flip."""
    result = pose_seq.copy()
    for left_idx, right_idx in COCO_KEYPOINT_FLIP_PAIRS:
        if left_idx < num_keypoints and right_idx < num_keypoints:
            result[:, left_idx], result[:, right_idx] = (
                pose_seq[:, right_idx].copy(),
                pose_seq[:, left_idx].copy()
            )
    return result


class PoseAugmentor:
    """
    Pose sequence augmentor with affine transforms.

    Based on STG-NF paper's augmentation strategy for pose-based
    anomaly detection.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        jitter_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        rotation_range: float = 10.0,
        shear_range: float = 0.1,
        translation_range: float = 0.1,
        temporal_dropout_prob: float = 0.1,
        keypoint_dropout_prob: float = 0.0,
        num_keypoints: int = 17
    ):
        """
        Initialize augmentor.

        Args:
            flip_prob: Probability of horizontal flip
            jitter_std: Standard deviation for coordinate jitter
            scale_range: (min, max) scale factors
            rotation_range: Max rotation in degrees
            shear_range: Max shear factor
            translation_range: Max translation factor
            temporal_dropout_prob: Probability of zeroing a frame
            keypoint_dropout_prob: Probability of zeroing a keypoint
            num_keypoints: Number of keypoints (17 for COCO)
        """
        self.flip_prob = flip_prob
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range
        self.temporal_dropout_prob = temporal_dropout_prob
        self.keypoint_dropout_prob = keypoint_dropout_prob
        self.num_keypoints = num_keypoints

    @classmethod
    def from_config(cls, config: Dict) -> 'PoseAugmentor':
        """Create augmentor from configuration dictionary."""
        aug_cfg = config.get('data', {}).get('augmentation', {})

        return cls(
            flip_prob=aug_cfg.get('flip_prob', 0.5),
            jitter_std=aug_cfg.get('jitter_std', 0.02),
            scale_range=tuple(aug_cfg.get('scale_range', [0.9, 1.1])),
            rotation_range=aug_cfg.get('rotation_range', 10.0),
            shear_range=aug_cfg.get('shear_range', 0.1),
            translation_range=aug_cfg.get('translation_range', 0.1),
            temporal_dropout_prob=aug_cfg.get('temporal_dropout_prob', 0.1),
            keypoint_dropout_prob=aug_cfg.get('keypoint_dropout_prob', 0.0),
            num_keypoints=config.get('model', {}).get('num_keypoints', 17)
        )

    def __call__(self, pose_seq: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to pose sequence.

        Args:
            pose_seq: Shape (T, V, C)

        Returns:
            Augmented pose sequence
        """
        result = pose_seq.copy()

        # Random augmentation parameters
        do_flip = np.random.random() < self.flip_prob
        shearx = np.random.uniform(-self.shear_range, self.shear_range)
        sheary = np.random.uniform(-self.shear_range, self.shear_range)
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        rotation = np.random.uniform(-self.rotation_range, self.rotation_range)
        tx = np.random.uniform(-self.translation_range, self.translation_range)
        ty = np.random.uniform(-self.translation_range, self.translation_range)

        # Build and apply transform
        transform = get_affine_transform_matrix(
            sx=scale, sy=scale,
            tx=tx, ty=ty,
            rot=rotation,
            shearx=shearx, sheary=sheary,
            flip=do_flip
        )
        result = apply_affine_transform(result, transform)

        # Swap keypoints if flipped
        if do_flip:
            result = flip_keypoints(result, self.num_keypoints)

        # Add coordinate jitter
        if self.jitter_std > 0:
            jitter = np.random.randn(*result[:, :, :2].shape) * self.jitter_std
            result[:, :, :2] += jitter

        # Temporal dropout
        if self.temporal_dropout_prob > 0:
            T = result.shape[0]
            for t in range(T):
                if np.random.random() < self.temporal_dropout_prob:
                    result[t] = 0

        # Keypoint dropout
        if self.keypoint_dropout_prob > 0:
            T, V, _ = result.shape
            for t in range(T):
                for v in range(V):
                    if np.random.random() < self.keypoint_dropout_prob:
                        result[t, v] = 0

        return result


class PoseLiftDataset(Dataset):
    """
    PyTorch Dataset for PoseLift pose sequences.

    Paper-aligned defaults:
    - seq_len=24 frames
    - stride=12 (50% overlap)
    - COCO-17 keypoints
    - 2 channels (x, y only)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        seq_len: int = 24,
        stride: int = 12,
        num_keypoints: int = 17,
        normalize: bool = True,
        include_confidence: bool = False,
        augmentor: Optional[PoseAugmentor] = None,
        max_gap: int = 5
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Path to PoseLift data directory
            split: 'train' or 'test'
            seq_len: Sequence length (paper: 24)
            stride: Sliding window stride (paper: 12)
            num_keypoints: Number of keypoints (17 for COCO)
            normalize: Whether to normalize coordinates
            include_confidence: Whether to include confidence channel
            augmentor: Optional PoseAugmentor for data augmentation
            max_gap: Maximum frame gap for continuity check
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.stride = stride
        self.num_keypoints = num_keypoints
        self.normalize = normalize
        self.include_confidence = include_confidence
        self.num_channels = 3 if include_confidence else 2
        self.augmentor = augmentor
        self.max_gap = max_gap

        self.samples: List[np.ndarray] = []
        self.labels: List[int] = []
        self.video_ids: List[str] = []
        self.frame_indices: List[List[int]] = []

        self._load_data()

    @classmethod
    def from_config(
        cls,
        config: Dict,
        split: str = 'train',
        augment: bool = True
    ) -> 'PoseLiftDataset':
        """
        Create dataset from configuration dictionary.

        Args:
            config: Configuration dictionary
            split: 'train' or 'test'
            augment: Whether to apply augmentation (only for training)

        Returns:
            Configured PoseLiftDataset
        """
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', {})
        aug_cfg = data_cfg.get('augmentation', {})

        augmentor = None
        if augment and split == 'train' and aug_cfg.get('enabled', True):
            augmentor = PoseAugmentor.from_config(config)

        return cls(
            data_dir=data_cfg.get('data_dir', '../shopformer/data/PoseLift'),
            split=split,
            seq_len=model_cfg.get('seq_len', 24),
            stride=data_cfg.get('stride', 12),
            num_keypoints=model_cfg.get('num_keypoints', 17),
            normalize=data_cfg.get('normalize', True),
            include_confidence=data_cfg.get('include_confidence', False),
            augmentor=augmentor
        )

    def _load_data(self):
        """Load and preprocess pose data from pickle files."""
        split_folder = 'Train' if self.split == 'train' else 'Test'
        pose_dir = self.data_dir / 'Pickle_files' / split_folder

        if not pose_dir.exists():
            raise FileNotFoundError(f"Pose directory not found: {pose_dir}")

        label_dir = self.data_dir / 'Pickle_files' / 'GT' if self.split == 'test' else None

        for pkl_file in sorted(pose_dir.glob('*.pkl')):
            video_name = pkl_file.stem

            with open(pkl_file, 'rb') as f:
                pose_data = pickle.load(f)

            frame_labels = None
            if label_dir is not None:
                label_file = label_dir / f'{video_name}.npy'
                if label_file.exists():
                    frame_labels = np.load(label_file)

            self._extract_sequences(pose_data, frame_labels, video_name)

        print(f"Loaded {len(self.samples)} sequences from {split_folder} split")
        if self.split == 'test':
            num_anomalies = sum(self.labels)
            print(f"  Normal: {len(self.labels) - num_anomalies}, Anomaly: {num_anomalies}")

    def _extract_sequences(
        self,
        pose_data: Dict,
        frame_labels: Optional[np.ndarray],
        video_name: str
    ):
        """
        Extract pose sequences using sliding window.

        Args:
            pose_data: Dictionary with frame data
            frame_labels: Optional ground truth labels
            video_name: Video identifier
        """
        # Organize poses by person
        person_poses: Dict[Any, Dict[int, Dict]] = {}

        for frame_num, frame_data in pose_data.items():
            if not frame_data or not isinstance(frame_data, dict):
                continue

            for person_id, person_data in frame_data.items():
                if not isinstance(person_data, (list, tuple)) or len(person_data) < 2:
                    continue

                bbox = person_data[0]
                keypoints = np.array(person_data[1])

                if keypoints is None:
                    continue
                if np.any(np.isnan(keypoints)) or np.any(np.isinf(keypoints)):
                    continue

                if person_id not in person_poses:
                    person_poses[person_id] = {}

                person_poses[person_id][int(frame_num)] = {
                    'keypoints': keypoints,
                    'bbox': np.array(bbox) if bbox is not None else None
                }

        # Extract sequences for each person
        for person_id, frames in person_poses.items():
            sorted_frames = sorted(frames.keys())

            if len(sorted_frames) < self.seq_len:
                continue

            for start_idx in range(0, len(sorted_frames) - self.seq_len + 1, self.stride):
                frame_indices = sorted_frames[start_idx:start_idx + self.seq_len]

                if not self._check_continuity(frame_indices):
                    continue

                pose_seq = self._extract_pose_sequence(frames, frame_indices)
                if pose_seq is None:
                    continue

                # Determine sequence label
                if frame_labels is not None:
                    seq_labels = [
                        frame_labels[min(f, len(frame_labels) - 1)]
                        for f in frame_indices
                    ]
                    # Majority voting for sequence label
                    label = 1 if sum(seq_labels) > len(seq_labels) // 2 else 0
                else:
                    label = 0  # Training data is normal

                self.samples.append(pose_seq)
                self.labels.append(label)
                self.video_ids.append(video_name)
                self.frame_indices.append(frame_indices)

    def _check_continuity(self, frame_indices: List[int]) -> bool:
        """Check if frame indices are reasonably continuous."""
        for i in range(1, len(frame_indices)):
            if frame_indices[i] - frame_indices[i - 1] > self.max_gap:
                return False
        return True

    def _extract_pose_sequence(
        self,
        frames: Dict,
        frame_indices: List[int]
    ) -> Optional[np.ndarray]:
        """Extract and preprocess pose sequence."""
        sequence = []

        for frame_idx in frame_indices:
            if frame_idx not in frames:
                return None

            keypoints = frames[frame_idx]['keypoints']

            # Handle different keypoint formats
            if keypoints.ndim == 1:
                keypoints = keypoints.reshape(-1, 3)
            elif keypoints.ndim == 2:
                keypoints = keypoints.copy()

            # Ensure we have at least 17 keypoints for COCO format
            if keypoints.shape[0] < 17:
                pad_size = 17 - keypoints.shape[0]
                keypoints = np.vstack([keypoints, np.zeros((pad_size, keypoints.shape[1]))])

            # Add synthetic neck keypoint if using 18 keypoints
            if self.num_keypoints == 18:
                keypoints = add_neck_keypoint(keypoints)
            else:
                keypoints = keypoints[:self.num_keypoints]

            # Extract channels
            if self.include_confidence:
                pose = keypoints[:, :3]
            else:
                pose = keypoints[:, :2]

            # Pad if necessary (shouldn't be needed after above processing)
            if len(pose) < self.num_keypoints:
                pad = np.zeros((self.num_keypoints - len(pose), self.num_channels))
                pose = np.vstack([pose, pad])

            sequence.append(pose)

        sequence = np.array(sequence)  # (T, V, C)

        if self.normalize:
            sequence = self._normalize_sequence(sequence)

        return sequence.astype(np.float32)

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normalize pose coordinates to [-1, 1] range.

        Centers on valid keypoints and scales by max extent.
        """
        coords = sequence[:, :, :2].copy()
        valid_mask = np.any(coords != 0, axis=-1)

        if valid_mask.sum() > 0:
            valid_coords = coords[valid_mask]
            center = valid_coords.mean(axis=0)
            centered = coords - center
            scale = np.abs(centered[valid_mask]).max() + 1e-6
        else:
            center = np.array([0.0, 0.0])
            scale = 1.0

        normalized = (coords - center) / scale
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        sequence[:, :, :2] = normalized

        return sequence

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Returns:
            poses: Shape (num_channels, seq_len, num_keypoints)
            label: 0 for normal, 1 for anomaly
        """
        pose = self.samples[idx].copy()

        # Apply augmentation (training only)
        if self.augmentor is not None:
            pose = self.augmentor(pose)

        # Transpose to (C, T, V) for PyTorch
        pose = np.transpose(pose, (2, 0, 1))

        return (
            torch.FloatTensor(pose),
            torch.LongTensor([self.labels[idx]]).squeeze()
        )

    def get_video_info(self, idx: int) -> Dict:
        """Get metadata for a sample."""
        return {
            'video_id': self.video_ids[idx],
            'frame_indices': self.frame_indices[idx],
            'label': self.labels[idx]
        }


class PoseLiftDataModule:
    """
    Data module for managing PoseLift dataset loading.

    Provides train and test DataLoaders with appropriate settings
    for MPS (Apple Silicon) compatibility.
    """

    def __init__(
        self,
        config: Dict,
        num_workers: int = 0  # MPS works best with num_workers=0
    ):
        """
        Initialize data module.

        Args:
            config: Configuration dictionary
            num_workers: Number of DataLoader workers (0 for MPS)
        """
        self.config = config
        self.num_workers = num_workers
        self.batch_size = config.get('training', {}).get('batch_size', 32)

        self.train_dataset: Optional[PoseLiftDataset] = None
        self.test_dataset: Optional[PoseLiftDataset] = None

    def setup(self):
        """Create train and test datasets."""
        self.train_dataset = PoseLiftDataset.from_config(
            self.config, split='train', augment=True
        )
        self.test_dataset = PoseLiftDataset.from_config(
            self.config, split='test', augment=False
        )

    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before getting dataloaders")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,  # MPS doesn't benefit from pin_memory
            drop_last=True
        )

    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before getting dataloaders")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False
        )

    def get_stats(self) -> Dict[str, int]:
        """Get dataset statistics."""
        stats = {}

        if self.train_dataset:
            stats['train_samples'] = len(self.train_dataset)

        if self.test_dataset:
            stats['test_samples'] = len(self.test_dataset)
            stats['test_normal'] = len(self.test_dataset.labels) - sum(self.test_dataset.labels)
            stats['test_anomaly'] = sum(self.test_dataset.labels)

        return stats
