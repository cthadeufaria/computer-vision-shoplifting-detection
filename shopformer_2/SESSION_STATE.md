# Shopformer_2 Session State
**Last Updated:** 2025-12-27
**Status:** Implementation plan ready, awaiting approval to proceed

---

## Current Todo List

| Task | Status | Notes |
|------|--------|-------|
| Update transformer to paper-optimal: 2 heads, 2 layers | Completed | Config updated |
| Add step LR scheduler: halve every 10 epochs | Completed | Changed to exponential (gamma=0.95) |
| Train model with paper-aligned configuration | Completed | AUC-ROC stuck at ~58% vs paper's 69% |
| Analyze training issues and implement fixes | In Progress | Plan created, ready for implementation |

---

## Problem Statement

Training achieves ~58% AUC-ROC instead of paper's 69%. The transformer gets best AUC-ROC at **epoch 1** and never improves thereafter. This suggests:
- Learning rate may be too high initially
- Loss-metric mismatch (MSE vs AUC-ROC)
- Possible token quality issues from GCAE

---

## Implementation Plan (6 Items)

### 1. Check GCAE Token Discriminability
**Status:** Not implemented
**Priority:** HIGH - Diagnostic step

Add `utils/diagnostics.py` with `analyze_token_discriminability()` function to measure:
- Token variance (information content)
- Inter-class distance (normal vs anomaly separation)
- Discriminability ratio

Run after Stage 1 training to validate GCAE quality before Stage 2.

### 2. Add Warmup (10 epochs, 1e-7 to 5e-5)
**Status:** Not implemented
**Priority:** HIGH - Most likely fix for "epoch 1 is best"

Config changes needed:
```yaml
scheduler:
  type: warmup_constant
  warmup_epochs: 10
  warmup_start_lr: 1.0e-7
  warmup_end_lr: 5.0e-5
```

Add `warmup_constant` scheduler type in `get_scheduler()` function in `train.py`.

### 3. Verify MSE Loss (Positionally-Encoded Tokens)
**Status:** Needs investigation
**Priority:** MEDIUM

Paper specifies MSE between:
- R̃ᵢᵗ⁰ (positionally-encoded tokens)
- R̂ᵢᵗ⁰ (reconstructed tokens)

Current implementation compares raw tokens (no PE) with reconstructed output.
May need to apply positional encoding before loss computation.

### 4. Early Stopping Based on Loss (Not AUC-ROC)
**Status:** Not implemented
**Priority:** MEDIUM

Current code (`train.py:372-395`) uses AUC-ROC for early stopping.
Paper trains for fixed 20 epochs with no early stopping.

Options:
- Switch to loss-based early stopping
- Disable early stopping entirely and train for 20 epochs

### 5. Learning Rate Rewinding
**Status:** Not implemented
**Priority:** LOW (try warmup first)

If warmup doesn't help, implement rewinding:
1. Train and track best epoch by loss
2. Reload best checkpoint
3. Fine-tune with 10x lower LR

### 6. Verify Training Only on Normal Data
**Status:** VERIFIED CORRECT

`poselift_dataset.py:477` confirms training uses only normal data:
```python
label = 0  # Training data is normal
```

---

## Key Files Modified

| File | Changes Made |
|------|--------------|
| `configs/paper_config.yaml` | 18 keypoints, 2 heads, 2 layers, 64 FFN, exponential LR |
| `train.py` | Adam optimizer, exponential scheduler, early stopping |
| `data/poselift_dataset.py` | Synthetic neck keypoint (18th) |
| `models/gcae.py` | 18-keypoint skeleton adjacency |
| `models/transformer.py` | Conditional projection layers |

---

## Paper Reference (arxiv:2504.19970)

Key specs from official Shopformer paper:
- **Optimal config:** 2 tokens, 144-dim embedding, 2 layers, 2 heads, FF=64
- **Training:** 20 epochs, Adam optimizer, LR 5e-5, no decay, no warmup
- **Data:** Train only on normal behavior, anomalies detected by reconstruction error
- **Loss:** MSE between positionally-encoded tokens and reconstructed tokens
- **Target AUC-ROC:** 69.15%

---

## Next Steps (When Resuming)

1. Read this file to restore context
2. Ask Claude to implement the 6-item plan above
3. Start with items 1 and 2 (diagnostics + warmup)
4. Run training and evaluate results

---

## Resume Command

When starting a new Claude session, use:
```
Read shopformer_2/SESSION_STATE.md and continue implementing the training fixes plan. Start with items 1 (token discriminability check) and 2 (warmup scheduler).
```
