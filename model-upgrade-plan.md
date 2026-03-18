# Model Upgrade Plan — STG-NF on PoseLift Dataset

## Context

The goal is to train STG-NF from scratch on the PoseLift shoplifting dataset and measure whether it outperforms the existing Shopformer_2 baseline (best AUC-ROC: **61.64%** on PoseLift). STG-NF achieves **85.9% AUC** on ShanghaiTech and **67.46% on PoseLift** (per paper) — but those were with the original ShanghaiTech-trained weights. Training directly on PoseLift with the correct adaptations should close the domain gap and produce a retail-specific model suitable for the iOS inference pipeline.

**Decision tree:**
1. Train STG-NF on PoseLift → evaluate AUC-ROC
2. If STG-NF ≥ Shopformer_2 (61.64%) → STG-NF becomes the primary model
3. If STG-NF < Shopformer_2 → further Shopformer_2 architecture work (see note at end)

---

## What STG-NF Needs to Train on PoseLift

STG-NF currently supports ShanghaiTech and UBnormal. PoseLift stores pose data differently — in pickle files, not the JSON sequences STG-NF expects. The following adaptations are required.

### Adaptation 1: PoseLift Dataset Loader

**File to create:** `stg_nf_official/datasets/poselift.py`

PoseLift format (from `shopformer_2/data/poselift_dataset.py`):
```
shopformer/data/PoseLift/Pickle_files/
├── Train/       # pickle files: frame_num → {person_id → [bbox, keypoints(17,3)]}
├── Test/        # same format
└── GT/          # ground truth: per-frame binary anomaly labels (test only)
```

The loader must:
1. Load pickle files from `Train/` and `Test/`
2. Extract 17 keypoints per person per frame
3. Apply `keypoints17_to_coco18()` (already in `stg_nf_official/dataset.py` — reuse it)
4. Build sliding windows of `seg_len=24` frames per track (match STG-NF default)
5. Normalize via `normalize_pose()` (already in `stg_nf_official/utils/data_utils.py` — reuse it)
6. Output shape: `[N, seg_len, 18, 3]` — same as existing STG-NF dataset classes
7. Load GT labels from `GT/` for test evaluation

**Reference:** `shopformer_2/data/poselift_dataset.py` — existing PoseLift loader in the repo. Use its pickle loading logic, but output in STG-NF's expected format instead of Shopformer's.

### Adaptation 2: Register PoseLift in `args.py`

**File to modify:** `stg_nf_official/args.py`

Add `'PoseLift'` as a valid `--dataset` choice alongside `'ShanghaiTech'` and `'UBnormal'`. Set dataset-specific defaults:

```python
# PoseLift dataset config
if args.dataset == 'PoseLift':
    args.seg_len = 24
    args.vid_res = [1920, 1080]   # default; override per-video if available
    args.data_dir = 'path/to/PoseLift/Pickle_files'
    args.ae_fn = None              # no appearance encoder needed
```

### Adaptation 3: Video Resolution Handling

STG-NF's `normalize_pose()` requires pixel-space coordinates and a `vid_res=[W, H]` to divide by. PoseLift pickles store raw keypoints — check whether they're already normalized (0–1) or in pixel space. If normalized, set `vid_res=[1,1]` for a no-op resolution divide.

**Check:** inspect a few pickle files from `shopformer/data/PoseLift/Pickle_files/Train/` to confirm coordinate range.

### Adaptation 4: Ground Truth Format Mapping

STG-NF evaluates against frame-level binary labels. PoseLift GT is stored per-frame in `GT/`. Write a converter in the dataset loader:
```python
def load_gt_labels(gt_dir, video_id) -> np.ndarray:
    # Returns binary array: 0=normal, 1=anomaly, shape=(num_frames,)
```

This feeds directly into STG-NF's existing `score_dataset()` evaluation pipeline in `stg_nf_official/utils/scoring_utils.py`.

---

## Training Run

Once adaptations are in place:

```bash
cd stg_nf_official

python train_eval.py \
  --dataset PoseLift \
  --data_dir ../shopformer/data/PoseLift/Pickle_files \
  --model_save_dir ../artifacts/stg_nf/poselift_runs \
  --epochs 50 \
  --seg_len 24 \
  --K 8 \
  --L 1 \
  --R 3.0 \
  --flow_coupling affine \
  --adj_strategy uniform \
  --max_hops 8 \
  --device mps
```

Log the run in `WORKLOG.md` after completion. Save trained weights to `artifacts/stg_nf/exports/PoseLift/`.

---

## Evaluation

Use STG-NF's existing evaluation pipeline — no new code needed:

```bash
python train_eval.py \
  --dataset PoseLift \
  --checkpoint ../artifacts/stg_nf/exports/PoseLift/<run_name>/weights.pt \
  --test_only
```

Metrics to record: **AUC-ROC** (primary), AUC-PR, optimal threshold.

Compare directly against Shopformer_2's best: `checkpoints/20251226_152230` — AUC-ROC **0.6164**.

---

## Implementation Order

| Step | Action | Gate |
|------|--------|------|
| 1 | Inspect PoseLift pickle format — confirm coordinate space | Know if vid_res=[1,1] or pixel |
| 2 | Write `stg_nf_official/datasets/poselift.py` | Dataset loads cleanly, shapes correct |
| 3 | Add PoseLift config to `stg_nf_official/args.py` | `--dataset PoseLift` accepted |
| 4 | Smoke test: load 1 batch, print shape | `[B, 24, 18, 3]` confirmed |
| 5 | Run training (50 epochs) | Loss decreasing, no NaN |
| 6 | Evaluate on PoseLift test split | AUC-ROC recorded |
| 7 | Compare vs Shopformer_2 baseline | Decision: proceed with STG-NF or pivot |
| 8 | If STG-NF wins: export model for iOS | `artifacts/stg_nf/exports/PoseLift/` |

---

## Critical Files

| File | Role |
|------|------|
| `stg_nf_official/dataset.py` | Reuse `keypoints17_to_coco18()` and `normalize_pose()` — do not duplicate |
| `stg_nf_official/utils/data_utils.py` | Reuse `normalize_pose()` |
| `stg_nf_official/utils/scoring_utils.py` | Existing AUC-ROC evaluation — no changes needed |
| `stg_nf_official/train_eval.py` | Training entry point — minimal changes to accept PoseLift |
| `stg_nf_official/args.py` | Add PoseLift dataset registration |
| `shopformer_2/data/poselift_dataset.py` | Reference for pickle loading logic |
| `WORKLOG.md` | Log every run |

---

## Verification

- Smoke test passes: `[B, 24, 18, 3]` batch loads from PoseLift pickles
- Training loss decreases monotonically for first 10 epochs (NLL decreasing = model learning)
- AUC-ROC on PoseLift test split computed and recorded in `WORKLOG.md`
- Weights saved to `artifacts/stg_nf/exports/PoseLift/<run>/weights.pt` with `model_args.json`

---

## If STG-NF Underperforms Shopformer_2

If the trained STG-NF AUC-ROC < 61.64% on PoseLift, the issue is likely one of:
- STG-NF's normalizing flow prior not suited to retail pose distributions (vs campus)
- Insufficient training data in PoseLift (153 clips) for the flow model to converge well

In that case, the path forward is improving Shopformer_2 — specifically: token discriminability diagnostics, ablations on seq_len/stride/embedding_dim, and adding a normalizing flow head to replace the MSE decoder. That work is documented separately and is out of scope for this plan.

---

---

## Sidenote: Future Research — DensePose from WiFi

> **For future exploration only — not part of current development roadmap**

**Paper:** *DensePose From WiFi* — Geng, Huang, De la Torre (2023)
**arXiv:** https://arxiv.org/abs/2301.00250

### What It Does

Maps WiFi signal characteristics (phase and amplitude from standard WiFi routers) through a deep neural network to **dense pose UV coordinates** across 24 body regions — achieving multi-person pose estimation with **no camera required**.

### Why It Matters for This Project

The current system's privacy advantage is pose-only detection (no RGB stored). WiFi-based DensePose would eliminate the camera entirely:

- No optical sensor → zero visual privacy risk → strongest possible GDPR position
- WiFi infrastructure already exists in retail stores (no new hardware)
- Works through walls, in low-light, from any angle — no blind spots
- Cannot be defeated by face coverings or occlusion

### Architecture Relevance

The output of WiFi DensePose (UV body region coordinates) is structurally compatible with the existing STG-NF and Shopformer pipelines. The keypoint extraction stage (`PoseEstimator` → `KeypointConverter`) would be replaced by a WiFi signal processing front-end; everything downstream (normalization, frame buffer, anomaly scoring) remains unchanged.

### Limitations to Investigate

- Requires dedicated WiFi sensing hardware or modified router firmware (not standard off-the-shelf)
- Performance in cluttered retail environments with many simultaneous WiFi devices is unknown
- Latency characteristics for real-time 24-frame windowed inference untested
- Current pose accuracy may be lower than Vision framework on clear camera footage

### Suggested Follow-up

1. Read full paper PDF for dataset details and quantitative pose accuracy metrics
2. Check if authors released code/model weights
3. Evaluate whether WiFi infrastructure costs + accuracy trade-off beats camera + privacy-risk mitigation costs for target retail customers
4. Consider as a premium "zero-camera" product tier once core camera-based system is validated
