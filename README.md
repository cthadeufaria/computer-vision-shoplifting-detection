# Computer Vision Shoplifting Detection

This repository is no longer a single pipeline. It contains several parallel
tracks for shoplifting and video-anomaly detection research:

- a legacy UCF-Crime tabular prototype in `legacy_ucf_crime/`
- a `shopformer/` PoseLift implementation with saved training artifacts
- a `shopformer_2/` PoseLift implementation that is the current main
  shoplifting path
- a fork-backed `stg_nf_official/` submodule for STG-NF with added workflow and
  video inference tooling

The previous root README described only the legacy prototype, which is not the
best entry point anymore.

Clone with submodules if you want the STG-NF code checked out immediately:

```bash
git clone --recurse-submodules git@github.com:cthadeufaria/computer-vision-shoplifting-detection.git
```

## Current Status

| Track | Dataset | Purpose | Status in repo |
| --- | --- | --- | --- |
| `legacy_ucf_crime/` | UCF-Crime | YOLO boxes -> CSV -> time-series classifier | Early prototype, incomplete, no saved metrics |
| `shopformer/` | PoseLift | First Shopformer implementation | Trained once; metrics saved |
| `shopformer_2/` | PoseLift | Main Shopformer development path | Training and evaluation code present; no saved metrics artifact committed yet |
| `stg_nf_official/` | ShanghaiTech, UBnormal, custom videos | STG-NF baseline, export, and video inference | Fork-backed submodule; benchmark tooling and custom inference code live there, while local artifacts live under `artifacts/stg_nf/` |

## Recorded Results In This Repo

These are the concrete result artifacts currently on disk:

- `shopformer/training_results.json`
  - PoseLift
  - AUC-ROC: `0.5701`
  - AUC-PR: `0.5361`
- `stg_nf_official/exports/ShanghaiTech/ShanghaiTech_85_9/metrics.json`
  - ShanghaiTech official checkpoint
  - AUC-ROC: `0.8594`
- `stg_nf_official/exports/UBnormal/UBnormal_supervised_79_2/metrics.json`
  - UBnormal supervised official checkpoint
  - AUC-ROC: `0.7916`
- `stg_nf_official/exports/UBnormal/UBnormal_unsupervised_71_8/metrics.json`
  - UBnormal unsupervised official checkpoint
  - AUC-ROC: `0.7178`

There is currently no committed `shopformer_2` evaluation artifact such as a
`metrics.json` file.

## Repository Map

### 1. `legacy_ucf_crime/`

Files:

- `legacy_ucf_crime/preprocess.py`
- `legacy_ucf_crime/dataset.py`
- `legacy_ucf_crime/model.py`
- `legacy_ucf_crime/train.py`
- `legacy_ucf_crime/main.py`

What it tries to do:

1. Read UCF-Crime shoplifting/shopping videos.
2. Track people with YOLO.
3. Save bounding boxes to CSV.
4. Train a time-series classifier on the tabular features.

Current state:

- useful as an early experiment only
- code is incomplete and not the recommended path
- no training or evaluation outputs are saved in the repo

Use this track only if you specifically want to continue the original
UCF-Crime tabular approach.

### 2. `shopformer/`

Purpose:

- first local implementation of the Shopformer paper idea on PoseLift

What exists:

- training and inference scripts
- multiple checkpoint folders
- one saved result artifact: `training_results.json`

Known result in repo:

- PoseLift AUC-ROC `0.5701`

### 3. `shopformer_2/`

Purpose:

- cleaner and more paper-aligned Shopformer implementation
- this is the main shoplifting-specific development path in the repo

What exists:

- two-stage training pipeline
  - Stage 1: GCAE tokenizer
  - Stage 2: Transformer reconstruction
- paper-aligned config in `shopformer_2/configs/paper_config.yaml`
- evaluation script with frame-level and optional video-level metrics
- diagnostics and device utilities

Current state:

- codebase is more mature than `shopformer/`
- project notes in `CLAUDE.md` and `WORKLOG.md` treat this as the recommended
  Shopformer path
- no committed evaluation artifact is present yet, so the repo does not contain
  a reproducible saved `shopformer_2` score to compare against `shopformer/`

### 4. `stg_nf_official/`

Purpose:

- local working copy of STG-NF with added usability improvements for this repo
- now backed by your fork at `cthadeufaria/STG-NF`

What exists in addition to the upstream-style training/eval code:

- `scripts/run_stg_nf.py`
  - wrapper for pretrained eval and training-from-scratch workflows
- `inference.py`
  - export-aware STG-NF inference helper
- `video_inference_pipeline.py`
  - raw video pipeline using MMPose pose estimation, IoU tracking, rolling
    STG-NF scoring, JSON outputs, and optional annotated video

What has already been done in this repo:

- official checkpoints were evaluated and exported
- benchmark metrics were saved under `stg_nf_official/exports/`
- raw-video inference tooling was added on top of pose-only STG-NF inputs
- sample video experiments were run and documented in
  `shoplifting-threshold-tuning.txt`

## STG-NF Development Stage In This Repo

The STG-NF work is past "just cloned the baseline" and is currently at the
"baseline integrated and custom inference built" stage.

Concretely, the repo already contains:

1. benchmark evaluation of official checkpoints on ShanghaiTech and UBnormal
2. exported inference artifacts for those checkpoints
3. a reusable inference helper (`stg_nf_official/inference.py`)
4. an end-to-end raw video pipeline (`stg_nf_official/video_inference_pipeline.py`)
   that adds:
   - MMPose-based pose extraction
   - IoU tracking
   - per-frame STG-NF scoring
   - JSON outputs
   - optional annotated video output

What is not done yet:

- no retail-specific training or fine-tuning checkpoint is committed
- no calibrated production threshold is established for real shoplifting footage
- current notes indicate a domain mismatch when using the ShanghaiTech model on
  retail-like sample videos: normal shopping footage can score as more anomalous
  than shoplifting footage

That means STG-NF in this repo is best described as:

- operational for benchmark evaluation
- operational for custom video scoring
- not yet validated as a retail-ready shoplifting detector

There is also one repo-management detail worth noting:

- the top-level repository currently sees `stg_nf_official` as changed because
  the nested STG-NF repo HEAD is ahead of the commit recorded by the parent
  repository

## Recommended Entry Points

Choose the path based on your goal:

- shoplifting research on PoseLift:
  - start in `shopformer_2/`
- STG-NF baseline comparison or custom pose/video anomaly scoring:
  - start in `stg_nf_official/`
- browser-based supervisory monitoring:
  - start in `web/`
- legacy UCF-Crime tabular experiment:
  - use `legacy_ucf_crime/`

## Quick Start By Track

### `web`

Purpose:

- browser version of the ShopliftDetect supervisory flow
- Smart Camera onboarding is intentionally blocked on web
- pairs with an iOS Smart Camera over local Wi-Fi using the displayed `sdlink://` payload

```bash
cd web
npm install
npm run dev
```

### `shopformer_2`

```bash
cd shopformer_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --config configs/paper_config.yaml
```

Evaluate a trained checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/stage2_best.pt --output results/
```

Expected dataset:

- PoseLift under the path configured by
  `shopformer_2/configs/paper_config.yaml`

### `stg_nf_official`

```bash
cd stg_nf_official
source .venv/bin/activate
python scripts/run_stg_nf.py --mode eval-pretrained --dataset shanghaitech
```

Run raw-video inference after exporting or selecting a model:

```bash
python video_inference_pipeline.py \
  --model ShanghaiTech_85_9 \
  --dataset ShanghaiTech \
  --video "../dataset/sinth/Shoplifting/Shoplifting (1).mp4" \
  --exports_root exports \
  --device mps \
  --pose_device cpu \
  --output_dir inference_outputs/video_online \
  --save_annotated_video \
  --display
```

Expected datasets:

- ShanghaiTech / UBnormal pose bundles for benchmark work
- optional custom videos for the raw-video pipeline

### `legacy_ucf_crime`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r legacy_ucf_crime/requirements.txt
python3 legacy_ucf_crime/preprocess.py
python3 legacy_ucf_crime/train.py
```

Expected dataset:

- UCF-Crime-style videos under `dataset/`

This path is not the recommended starting point unless you intend to repair and
continue the tabular pipeline.

## Supporting Notes

- `WORKLOG.md` tracks the high-level experiment inventory.
- `CLAUDE.md` records the latest project context and recent STG-NF threshold
  tuning notes.
- `shoplifting-threshold-tuning.txt` contains the recent STG-NF inference
  session transcript for Sinth retail sample videos.
