# Shoplifting Detection - Planning & Worklog
**Last Updated:** 2026-01-29

This file is the single source of truth for current status, model results, and next actions.

---

## 1) Repo Map (Current Tracks)

1. **Legacy tabular pipeline (root)**  
   Files: `preprocess.py`, `dataset.py`, `train.py`, `model.py`  
   Dataset: UCF-Crime videos -> tabular CSV features  
   Status: No metrics/results artifacts found in repo.

2. **Shopformer (pose-based, v1)**  
   Folder: `shopformer/`  
   Dataset: PoseLift (pose sequences)  
   Status: Has one recorded results file: `shopformer/training_results.json`.

3. **Shopformer_2 (pose-based, v2 / main path)**  
   Folder: `shopformer_2/`  
   Dataset: PoseLift  
   Status: No saved metrics/results artifacts in repo; training/eval code exists.

---

## 2) Results Inventory (What Exists Right Now)

Only one concrete result artifact is present in the repo.

| Model | File | Dataset | AUC-ROC | AUC-PR | Notes |
| --- | --- | --- | --- | --- | --- |
| Shopformer (v1) | `shopformer/training_results.json` | PoseLift | **0.5701** | 0.5361 | 629 test samples; best epoch 3 |
| Shopformer_2 | _None found_ | PoseLift | _N/A_ | _N/A_ | Run evaluation to generate `metrics.json` |
| Legacy tabular | _None found_ | UCF-Crime | _N/A_ | _N/A_ | No artifacts logged |

### Shopformer (v1) - Key Metrics Snapshot
- AUC-ROC: 0.5701  
- AUC-PR: 0.5361  
- Accuracy: 0.5644  
- Precision: 0.5371  
- Recall: 0.7785  
- F1: 0.6356  
- Optimal threshold: 0.1403  

---

## 3) Model Comparison (How to Compare Apples-to-Apples)

To compare models meaningfully, align:
- **Dataset:** PoseLift for shopformer-based models (do not compare to UCF-Crime runs).
- **Split & preprocessing:** Ensure identical pose preprocessing.
- **Metric:** AUC-ROC primary, plus AUC-PR and FPR at target operating points.

### Current Gap
- Only Shopformer (v1) has an on-disk metrics artifact.
- Shopformer_2 evaluation outputs `metrics.json` but no run has been saved yet.

---

## 4) Starting Point (Recommended Next Actions)

1. **Pick a single path to advance (recommended: `shopformer_2/`).**  
2. **Run evaluation and save metrics:**
   - Train: `cd shopformer_2 && python train.py --config configs/paper_config.yaml`
   - Evaluate: `python evaluate.py --checkpoint checkpoints/stage2_best.pt --output results/`
3. **Log results** in this file using the template below.
4. **Run diagnostics** to confirm tokenizer quality:
   - `python -c "from utils.diagnostics import run_full_diagnostics; ..."`

---

## 5) Worklog Entries

Use this section for concise, chronological updates.

### 2026-01-29
- Baseline inventory completed. Only `shopformer/training_results.json` has recorded metrics.
- Shopformer (v1) best AUC-ROC: 0.5701 on PoseLift.
- Shopformer_2 has evaluation code but no saved results artifact yet.

---

## 6) Experiment Tracking Template

Copy/paste for each run:

```
### YYYY-MM-DD - <short run name>
- Model: shopformer | shopformer_2 | legacy tabular
- Dataset: PoseLift | UCF-Crime | other
- Config: <file / key hyperparams>
- Checkpoint: <path>
- Metrics:
  - AUC-ROC:
  - AUC-PR:
  - F1:
  - FPR @ <threshold>:
- Notes:
  - <anything notable>
```

---

## 7) Open Questions / Gaps

- Where are the PoseLift data and checkpoints stored locally?
- Do you want to preserve compatibility with the original `shopformer/` pipeline?
- Should we standardize a single output folder (e.g., `results/`) at repo root?

