#!/usr/bin/env python3
"""
Generate test fixture JSON files for the iOS Swift unit tests.

Usage (from repo root):
    python3 scripts/generate_fixtures.py

Outputs:
    ios/ShopliftDetectTests/Fixtures/coco17_sample.json
    ios/ShopliftDetectTests/Fixtures/normal_pose_window.json
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STG_NF_DIR = os.path.join(REPO_ROOT, "stg_nf_official")
FIXTURES_DIR = os.path.join(REPO_ROOT, "ios", "ShopliftDetectTests", "Fixtures")
sys.path.insert(0, STG_NF_DIR)

from dataset import keypoints17_to_coco18          # noqa: E402
from utils.data_utils import normalize_pose         # noqa: E402


# ---------------------------------------------------------------------------
# Fixture 1: COCO17 → COCO18 keypoint conversion
# ---------------------------------------------------------------------------
def generate_coco17_fixture():
    """
    Input: single frame, single person — 17 keypoints (x, y, conf).
    Interesting values: nose (idx 0), left shoulder (idx 5), right shoulder (idx 6).
    Everything else is zero.

    keypoints17_to_coco18 expects shape [..., 17, 3].
    We pass [1, 1, 17, 3] (N=1, T=1, V=17, F=3) so the ellipsis covers (N, T).
    """
    kps = np.zeros((1, 1, 17, 3), dtype=np.float32)

    # COCO17 joint indices:
    #   0 = nose, 5 = leftShoulder, 6 = rightShoulder
    kps[0, 0, 0]  = [100.0, 200.0, 0.90]   # nose
    kps[0, 0, 5]  = [150.0, 300.0, 0.85]   # leftShoulder
    kps[0, 0, 6]  = [250.0, 300.0, 0.88]   # rightShoulder

    # Synthetic neck = average of shoulders (used when Vision neck confidence < 0.3)
    expected_neck = 0.5 * (kps[0, 0, 5] + kps[0, 0, 6])   # [200, 300, 0.865]

    # Run the Python reference function
    coco18 = keypoints17_to_coco18(kps)  # [1, 1, 18, 3]

    # opp_order maps the 18-element array to OpenPose order.
    # After reindexing: index 1 = neck (was appended at idx 17, now reordered to 1)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

    fixture = {
        "description": "Single frame, one person. nose+shoulders filled, rest zero.",
        "input_coco17": kps[0, 0].tolist(),          # [17, 3]
        "output_coco18": coco18[0, 0].tolist(),       # [18, 3]
        "opp_order": opp_order,
        "expected_neck_xy": expected_neck[:2].tolist(),
        "expected_neck_conf": float(expected_neck[2]),
        # Spot-checks (0-indexed into output_coco18):
        "checks": {
            "neck_index_in_output": 1,           # opp_order[1] = 17 = neck
            "nose_index_in_output": 0,           # opp_order[0] = 0 = nose
            # opp_order[2] = 6 = rightShoulder
            "right_shoulder_index_in_output": 2,
            # opp_order[5] = 5 = leftShoulder
            "left_shoulder_index_in_output": 5,
        },
    }

    path = os.path.join(FIXTURES_DIR, "coco17_sample.json")
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"✓ {path}")
    print(f"  neck in output[1]: {coco18[0, 0, 1].tolist()}")
    print(f"  nose in output[0]: {coco18[0, 0, 0].tolist()}")


# ---------------------------------------------------------------------------
# Fixture 2: pose normalization
# ---------------------------------------------------------------------------
def generate_normalization_fixture():
    """
    Input: seeded random 24-frame × 18-joint window, shape [1, 24, 18, 3].
    vid_res = [640, 480].

    normalize_pose live branch (data_utils.py line 156):
        1. pose_data / [W, H, 1]           -- divide x by W, y by H, conf unchanged
        2. subtract mean of xy over axes (1, 2) (T and V dimensions)
        3. divide xy by std of original (pre-subtraction) y over axes (1, 2)

    The Swift PoseNormalizer must reproduce this exactly.
    """
    rng = np.random.RandomState(42)
    # Random xy in pixel range; confidences between 0.3 and 1.0
    window = rng.randn(1, 24, 18, 3).astype(np.float32)
    # Scale xy to plausible pixel range
    window[..., 0] = window[..., 0] * 100 + 320   # x around 320 ± 100
    window[..., 1] = window[..., 1] * 60  + 240   # y around 240 ± 60
    window[..., 2] = np.clip(np.abs(rng.randn(1, 24, 18).astype(np.float32)), 0.3, 1.0)

    vid_res = [640, 480]
    normalized = normalize_pose(window.copy(), vid_res=vid_res)

    fixture = {
        "description": "Seeded (seed=42) 24-frame × 18-joint window. vid_res=[640,480].",
        "vid_res": vid_res,
        "input": window.tolist(),            # [1, 24, 18, 3]
        "expected_output": normalized.tolist(),   # [1, 24, 18, 3]
        # Intermediate values for debugging Swift implementation
        "debug": {
            "mean_x": float(window[0, :, :, 0].mean() / vid_res[0]),
            "mean_y": float(window[0, :, :, 1].mean() / vid_res[1]),
            "std_y_before_subtraction": float(
                (window[0, :, :, 1] / vid_res[1]).std()
            ),
        },
    }

    path = os.path.join(FIXTURES_DIR, "normal_pose_window.json")
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"✓ {path}")
    print(f"  input  shape: {window.shape}")
    print(f"  output shape: {normalized.shape}")
    print(f"  output xy mean (should be ~0): {normalized[0, :, :, :2].mean():.6f}")
    print(f"  output y  std  (should be ~1): {normalized[0, :, :, 1].std():.6f}")


# ---------------------------------------------------------------------------
# Fixture 3: PyTorch NLL reference (optional — requires torch + checkpoint)
# ---------------------------------------------------------------------------
def generate_nll_fixture():
    """
    Requires Python 3.11 with torch installed and ShanghaiTech_85_9.tar checkpoint.
    Generates inference_nll_sample.json with a seeded [1,2,24,18] input and PyTorch NLL.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        print("⚠  torch not available — skipping NLL fixture (re-run with Python 3.11)")
        return

    SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, SCRIPTS_DIR)
    try:
        from convert_stgnf_to_coreml import build_model  # noqa: F401, E402
    except Exception as e:
        print(f"⚠  Could not import conversion script ({e}) — skipping NLL fixture")
        return

    print("→ Building STG-NF model for NLL fixture …")
    try:
        wrapper = build_model()
    except Exception as e:
        print(f"⚠  build_model() failed ({e}) — skipping NLL fixture")
        return

    import torch as torch_module
    torch_module.manual_seed(0)
    example = torch_module.randn(1, 2, 24, 18)

    with torch_module.no_grad():
        nll_tensor = wrapper(example)
    nll_value = float(nll_tensor.item())

    fixture = {
        "description": "Seeded (seed=0) random input [1,2,24,18]. PyTorch NLL reference.",
        "input_pose_window": example.numpy().tolist(),  # [1,2,24,18]
        "expected_nll": nll_value,
    }

    path = os.path.join(FIXTURES_DIR, "inference_nll_sample.json")
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"✓ {path}")
    print(f"  NLL = {nll_value:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    print("Generating fixtures …\n")
    generate_coco17_fixture()
    print()
    generate_normalization_fixture()
    print()
    generate_nll_fixture()
    print("\nDone.")
