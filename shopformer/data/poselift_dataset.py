"""
PoseLift Dataset Loader

This module implements data loading for the PoseLift dataset,
the benchmark dataset for pose-based shoplifting detection.

Dataset Structure (downloaded from official source):
    DATA/PoseLift/
    ├── Pickle_files/
    │   ├── Train/
    │   │   └── <camera>_<video>.pkl
    │   ├── Test/
    │   │   └── <camera>_<video>.pkl
    │   └── GT/
    │       └── <camera>_<video>.npy
    └── ...

Each .pkl file contains:
    - Dictionary with frame numbers as keys
    - Each frame contains: person_id, bbox (XYWH), keypoints (XYC format)

Each .npy label file contains:
    - Binary array (0=normal, 1=shoplifting)
"""

import os
import pickle
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path


# ============================================================================
# Affine Transform Augmentations (based on STG-NF paper approach)
# ============================================================================

def get_affine_transform_matrix(sx=1, sy=1, tx=0, ty=0, rot=0, shearx=0., sheary=0., flip=False):
    """Generate 3x3 affine transformation matrix."""
    cos_r = math.cos(math.radians(rot))
    sin_r = math.sin(math.radians(rot))

    # Flip matrix
    flip_val = -1.0 if flip else 1.0

    # Combined transformation
    # Order: flip -> scale/translate -> shear -> rotate
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


def apply_affine_transform(pose_seq, transform_matrix):
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

    # Transform x, y coordinates
    coords = pose_seq[:, :, :2]  # (T, V, 2)
    ones = np.ones((T, V, 1))
    coords_h = np.concatenate([coords, ones], axis=-1)  # (T, V, 3)

    # Apply transformation: (T, V, 3) @ (3, 3).T -> (T, V, 3)
    transformed = np.einsum('tvc,cd->tvd', coords_h, transform_matrix[:2, :].T)

    result[:, :, :2] = transformed
    return result


# COCO keypoint indices for left-right swap
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


def flip_keypoints(pose_seq, num_keypoints=17):
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
    Based on STG-NF paper's augmentation strategy.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        shear_range: float = 0.1,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        rotation_range: float = 10.0,  # degrees
        translation_range: float = 0.1,
        jitter_std: float = 0.02,
        temporal_dropout_prob: float = 0.1,
        num_keypoints: int = 17
    ):
        self.flip_prob = flip_prob
        self.shear_range = shear_range
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.jitter_std = jitter_std
        self.temporal_dropout_prob = temporal_dropout_prob
        self.num_keypoints = num_keypoints

    def __call__(self, pose_seq: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to pose sequence.

        Args:
            pose_seq: Shape (T, V, C)

        Returns:
            Augmented pose sequence
        """
        result = pose_seq.copy()

        # Random horizontal flip
        do_flip = np.random.random() < self.flip_prob

        # Random shear
        shearx = np.random.uniform(-self.shear_range, self.shear_range)
        sheary = np.random.uniform(-self.shear_range, self.shear_range)

        # Random scale
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Random rotation
        rotation = np.random.uniform(-self.rotation_range, self.rotation_range)

        # Random translation
        tx = np.random.uniform(-self.translation_range, self.translation_range)
        ty = np.random.uniform(-self.translation_range, self.translation_range)

        # Build transform matrix
        transform = get_affine_transform_matrix(
            sx=scale, sy=scale,
            tx=tx, ty=ty,
            rot=rotation,
            shearx=shearx, sheary=sheary,
            flip=do_flip
        )

        # Apply affine transform
        result = apply_affine_transform(result, transform)

        # Swap left-right keypoints if flipped
        if do_flip:
            result = flip_keypoints(result, self.num_keypoints)

        # Add coordinate jitter
        if self.jitter_std > 0:
            jitter = np.random.randn(*result[:, :, :2].shape) * self.jitter_std
            result[:, :, :2] += jitter

        # Temporal dropout (zero out random frames)
        if self.temporal_dropout_prob > 0:
            T = result.shape[0]
            for t in range(T):
                if np.random.random() < self.temporal_dropout_prob:
                    result[t] = 0

        return result


class PoseLiftDataset(Dataset):
    """
    PyTorch Dataset for PoseLift pose sequences.

    Extracts sliding window segments from pose sequences for
    training and evaluation.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        seq_len: int = 12,
        stride: int = 6,
        num_keypoints: int = 17,
        normalize: bool = True,
        include_confidence: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.stride = stride
        self.num_keypoints = num_keypoints
        self.normalize = normalize
        self.include_confidence = include_confidence
        self.num_channels = 3 if include_confidence else 2

        self.samples = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        """Load and preprocess pose data."""
        # Map split names to PoseLift folder names
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

    def _extract_sequences(
        self,
        pose_data: Dict,
        frame_labels: Optional[np.ndarray],
        video_name: str
    ):
        """Extract pose sequences using sliding window.

        Handles PoseLift pickle format:
            {frame_num: {person_id: [bbox, keypoints_array]}}
        """
        person_poses = {}

        for frame_num, frame_data in pose_data.items():
            # Skip empty frames
            if not frame_data or not isinstance(frame_data, dict):
                continue

            # frame_data is {person_id: [bbox, keypoints]}
            for person_id, person_data in frame_data.items():
                # person_data is [bbox, keypoints] where keypoints is (17, 3) array
                if not isinstance(person_data, (list, tuple)) or len(person_data) < 2:
                    continue

                bbox = person_data[0]
                keypoints = np.array(person_data[1])

                # Skip if keypoints is None or contains NaN/inf
                if keypoints is None:
                    continue
                if np.any(np.isnan(keypoints)) or np.any(np.isinf(keypoints)):
                    continue

                if person_id not in person_poses:
                    person_poses[person_id] = {}

                person_poses[person_id][int(frame_num)] = {
                    'keypoints': keypoints,  # already np.array from above
                    'bbox': np.array(bbox) if bbox is not None else None
                }

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

                if frame_labels is not None:
                    seq_labels = [
                        frame_labels[min(f, len(frame_labels) - 1)]
                        for f in frame_indices
                    ]
                    label = 1 if sum(seq_labels) > len(seq_labels) // 2 else 0
                else:
                    label = 0

                self.samples.append(pose_seq)
                self.labels.append(label)

    def _check_continuity(self, frame_indices: List[int], max_gap: int = 5) -> bool:
        for i in range(1, len(frame_indices)):
            if frame_indices[i] - frame_indices[i - 1] > max_gap:
                return False
        return True

    def _extract_pose_sequence(
        self,
        frames: Dict,
        frame_indices: List[int]
    ) -> Optional[np.ndarray]:
        sequence = []

        for frame_idx in frame_indices:
            if frame_idx not in frames:
                return None

            keypoints = frames[frame_idx]['keypoints']

            if keypoints.ndim == 1:
                keypoints = keypoints.reshape(-1, 3)[:self.num_keypoints]
            elif keypoints.ndim == 2:
                keypoints = keypoints[:self.num_keypoints]

            if self.include_confidence:
                pose = keypoints[:, :3]
            else:
                pose = keypoints[:, :2]

            if len(pose) < self.num_keypoints:
                pad = np.zeros((self.num_keypoints - len(pose), self.num_channels))
                pose = np.vstack([pose, pad])

            sequence.append(pose)

        sequence = np.array(sequence)

        if self.normalize:
            sequence = self._normalize_sequence(sequence)

        return sequence.astype(np.float32)

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize pose coordinates to [-1, 1] range."""
        coords = sequence[:, :, :2].copy()

        # Find valid (non-zero) coordinates
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

        # Ensure no NaN/inf in output
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        sequence[:, :, :2] = normalized
        return sequence

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pose = self.samples[idx]
        pose = np.transpose(pose, (2, 0, 1))

        return (
            torch.FloatTensor(pose),
            torch.LongTensor([self.labels[idx]]).squeeze()
        )


class SyntheticPoseLiftDataset(Dataset):
    """
    Synthetic dataset for testing Shopformer when PoseLift is unavailable.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 12,
        num_keypoints: int = 17,
        num_channels: int = 2,
        anomaly_ratio: float = 0.3
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_keypoints = num_keypoints
        self.num_channels = num_channels

        self.samples = []
        self.labels = []

        for i in range(num_samples):
            is_anomaly = np.random.random() < anomaly_ratio
            base_pose = self._generate_skeleton()
            sequence = self._generate_sequence(base_pose, is_anomaly)
            self.samples.append(sequence)
            self.labels.append(1 if is_anomaly else 0)

    def _generate_skeleton(self) -> np.ndarray:
        skeleton = np.array([
            [0.5, 0.1], [0.48, 0.08], [0.52, 0.08], [0.45, 0.1], [0.55, 0.1],
            [0.4, 0.25], [0.6, 0.25], [0.35, 0.4], [0.65, 0.4], [0.3, 0.55],
            [0.7, 0.55], [0.45, 0.55], [0.55, 0.55], [0.43, 0.75], [0.57, 0.75],
            [0.42, 0.95], [0.58, 0.95],
        ])
        skeleton += np.random.randn(*skeleton.shape) * 0.02
        return skeleton[:self.num_keypoints]

    def _generate_sequence(self, base_pose: np.ndarray, is_anomaly: bool) -> np.ndarray:
        sequence = []
        for t in range(self.seq_len):
            pose = base_pose.copy()
            motion_scale = 0.02 if not is_anomaly else 0.08
            pose += np.random.randn(*pose.shape) * motion_scale

            if is_anomaly and t > self.seq_len // 2:
                pose[9] = pose[9] * 0.7 + pose[11] * 0.3
                pose[10] = pose[10] * 0.7 + pose[12] * 0.3

            sequence.append(pose)
        return np.array(sequence)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pose = self.samples[idx]
        pose = np.transpose(pose, (2, 0, 1))
        return (
            torch.FloatTensor(pose),
            torch.LongTensor([self.labels[idx]]).squeeze()
        )


class PoseLiftDataModule:
    """Data module for managing PoseLift dataset loading."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        seq_len: int = 12,
        stride: int = 6,
        num_workers: int = 4,
        use_synthetic: bool = False,
        synthetic_samples: int = 1000
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride = stride
        self.num_workers = num_workers
        self.use_synthetic = use_synthetic
        self.synthetic_samples = synthetic_samples

        self.train_dataset = None
        self.test_dataset = None

    def setup(self):
        if self.use_synthetic:
            self.train_dataset = SyntheticPoseLiftDataset(
                num_samples=self.synthetic_samples,
                seq_len=self.seq_len,
                anomaly_ratio=0.0
            )
            self.test_dataset = SyntheticPoseLiftDataset(
                num_samples=self.synthetic_samples // 5,
                seq_len=self.seq_len,
                anomaly_ratio=0.3
            )
        else:
            self.train_dataset = PoseLiftDataset(
                data_dir=self.data_dir,
                split='train',
                seq_len=self.seq_len,
                stride=self.stride
            )
            self.test_dataset = PoseLiftDataset(
                data_dir=self.data_dir,
                split='test',
                seq_len=self.seq_len,
                stride=self.stride
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
