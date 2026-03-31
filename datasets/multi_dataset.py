"""
Multi-dataset combiner for STG-NF training.

Combines any number of PoseLiftDataset and PoseSegDataset instances into a single
ConcatDataset.  Each constituent dataset applies its own normalization internally,
so cross-dataset resolution differences are handled before concatenation.

The resulting MultiDataset can be used as a drop-in replacement for a single
PoseLiftDataset in the training loop.

Public API
----------
    MultiDataset   – torch.utils.data.ConcatDataset with a .metadata attribute
    build_multi_train_dataset(args) -> MultiDataset
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch.utils.data

# Allow importing from stg_nf_official/ regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_STGNF_DIR = _REPO_ROOT / "stg_nf_official"
if str(_STGNF_DIR) not in sys.path:
    sys.path.insert(0, str(_STGNF_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.poselift_stgnf import PoseLiftDataset
from dataset import PoseSegDataset


class _Float32PoseSegDataset(torch.utils.data.Dataset):
    """
    Thin wrapper around PoseSegDataset that casts every numpy array in the
    returned item to float32.

    PoseSegDataset.labels defaults to np.ones(...) which is float64, and
    normalize_pose can produce float64 when dividing float32 data by an int64
    norm_factor.  MPS (Apple Silicon) rejects float64 tensors, so we cast
    before the DataLoader collates items into batches.
    """

    def __init__(self, dataset: PoseSegDataset):
        self._ds = dataset
        self.metadata = dataset.metadata

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx):
        item = self._ds[idx]
        out = []
        for x in item:
            if isinstance(x, np.ndarray):
                out.append(x.astype(np.float32))
            elif isinstance(x, np.floating):  # numpy scalar (e.g. labels[i] from np.ones)
                out.append(np.float32(x))
            else:
                out.append(x)
        return out


class MultiDataset(torch.utils.data.ConcatDataset):
    """
    Concatenation of multiple pose datasets for multi-source STG-NF training.

    Attributes
    ----------
    metadata : np.ndarray  [N_total, 4]
        Concatenated [scene_id, clip_id, person_id, start_frame] from all
        constituent datasets.  Used only for diagnostics during training;
        evaluation always runs on the separate PoseLift test dataset.
    """

    def __init__(self, datasets):
        if not datasets:
            raise ValueError("MultiDataset: at least one dataset is required")
        super().__init__(datasets)
        # Metadata dtypes differ across loaders (PoseLiftDataset: int64,
        # PoseSegDataset: object/str for scene/clip IDs from JSON filenames).
        # Cast to object so np.concatenate handles mixed types safely.
        # Note: this metadata is used for diagnostics only — evaluation always
        # uses dataset["test"].metadata from the separate PoseLift test split.
        self.metadata = np.concatenate(
            [d.metadata.astype(object) for d in datasets], axis=0
        )

    def __repr__(self) -> str:
        sizes = [len(d) for d in self.datasets]
        names = [type(d).__name__ for d in self.datasets]
        parts = ", ".join(f"{n}({s})" for n, s in zip(names, sizes))
        return f"MultiDataset(total={len(self)}, [{parts}])"


def _dir_has_pickles(path: str) -> bool:
    """Return True if path is an existing directory containing at least one .pkl file."""
    if not os.path.isdir(path):
        return False
    return any(f.endswith(".pkl") for f in os.listdir(path))


def build_multi_train_dataset(args) -> MultiDataset:
    """
    Instantiate all available training datasets and concatenate them.

    Datasets included (each only if the data directory exists and is non-empty):
    - PoseLift       – pickle files from args.pickle_path['train']        vid_res=[1024,1440]
    - sinth Normal   – pickle files from args.sinth_data_dir/Normal       vid_res=[1920,1080]
    - ShanghaiTech   – AlphaPose JSONs from args.shanghaitech_pose_dir    vid_res=[856,480]
    - UBnormal       – AlphaPose JSONs from args.ubnormal_pose_dir        vid_res=[1280,720]

    Parameters
    ----------
    args : argparse.Namespace
        Must have: pickle_path['train'], seg_len, seg_stride,
                   sinth_data_dir, shanghaitech_pose_dir, ubnormal_pose_dir
    """
    datasets = []

    # --- PoseLift (always required) ----------------------------------------
    poselift_train_dir = args.pickle_path["train"]
    if not _dir_has_pickles(poselift_train_dir):
        raise RuntimeError(
            f"PoseLift train pickles not found at: {poselift_train_dir}\n"
            "Check --data_dir points to the directory containing PoseLift/Pickle_files/."
        )
    print(f"[MultiDataset] Loading PoseLift from {poselift_train_dir} ...")
    datasets.append(
        PoseLiftDataset(
            pickle_dir=poselift_train_dir,
            seg_len=args.seg_len,
            stride=args.seg_stride,
            vid_res=[1024, 1440],
        )
    )
    print(f"[MultiDataset]   PoseLift: {len(datasets[-1])} segments")

    # --- sinth Normal (optional, skip if extraction hasn't been run yet) ----
    sinth_dir = os.path.join(args.sinth_data_dir, "Normal")
    if _dir_has_pickles(sinth_dir):
        print(f"[MultiDataset] Loading sinth Normal from {sinth_dir} ...")
        datasets.append(
            PoseLiftDataset(
                pickle_dir=sinth_dir,
                seg_len=args.seg_len,
                stride=args.seg_stride,
                vid_res=[1920, 1080],
            )
        )
        print(f"[MultiDataset]   sinth Normal: {len(datasets[-1])} segments")
    else:
        print(
            f"[MultiDataset] sinth Normal skipped (no pickles at {sinth_dir}).\n"
            "  Run: python scripts/extract_poses_from_videos.py "
            "--input_dir data/sinth/Normal --output_dir data/sinth/Pickle_files/Normal"
        )

    # --- ShanghaiTech (JSON, native PoseSegDataset) -------------------------
    st_dir = args.shanghaitech_pose_dir
    if os.path.isdir(st_dir) and os.listdir(st_dir):
        print(f"[MultiDataset] Loading ShanghaiTech from {st_dir} ...")
        datasets.append(
            _Float32PoseSegDataset(
                PoseSegDataset(
                    path_to_json_dir=st_dir,
                    normalize_pose_segs=True,
                    seg_len=args.seg_len,
                    seg_stride=args.seg_stride,
                    trans_list=None,       # disable aug multiplier to keep dataset sizes comparable
                    dataset="ShanghaiTech",
                    vid_res=[856, 480],
                    train_seg_conf_th=0.0,
                )
            )
        )
        print(f"[MultiDataset]   ShanghaiTech: {len(datasets[-1])} segments")
    else:
        print(f"[MultiDataset] ShanghaiTech skipped (not found at {st_dir})")

    # --- UBnormal (JSON, native PoseSegDataset) -----------------------------
    ub_dir = args.ubnormal_pose_dir
    if os.path.isdir(ub_dir) and os.listdir(ub_dir):
        print(f"[MultiDataset] Loading UBnormal from {ub_dir} ...")
        datasets.append(
            _Float32PoseSegDataset(
                PoseSegDataset(
                    path_to_json_dir=ub_dir,
                    normalize_pose_segs=True,
                    seg_len=args.seg_len,
                    seg_stride=args.seg_stride,
                    trans_list=None,
                    dataset="UBnormal",
                    vid_res=[1280, 720],
                    train_seg_conf_th=0.0,
                )
            )
        )
        print(f"[MultiDataset]   UBnormal: {len(datasets[-1])} segments")
    else:
        print(f"[MultiDataset] UBnormal skipped (not found at {ub_dir})")

    multi = MultiDataset(datasets)
    print(f"[MultiDataset] Total training segments: {len(multi)}")
    return multi
