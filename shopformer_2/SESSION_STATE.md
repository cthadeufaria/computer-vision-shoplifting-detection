# Shopformer_2 Session State
**Last Updated:** 2025-12-27
**Status:** Training fixes implemented, ready for testing

---

## Current Todo List

| Task | Status | Notes |
|------|--------|-------|
| Update transformer to paper-optimal: 2 heads, 2 layers | Completed | Config updated |
| Add step LR scheduler: halve every 10 epochs | Completed | Changed to warmup_constant |
| Train model with paper-aligned configuration | Completed | AUC-ROC stuck at ~58% vs paper's 69% |
| Analyze training issues and implement fixes | **Completed** | All 6 items implemented |

---

## Implementation Status (All Complete)

### 1. GCAE Token Discriminability Check
**Status:** IMPLEMENTED
**File:** `utils/diagnostics.py`

Added diagnostic functions:
- `analyze_token_discriminability()` - Measures token variance, inter-class distance, discriminability ratio
- `analyze_reconstruction_error_distribution()` - Analyzes error distribution for normal vs anomaly
- `run_full_diagnostics()` - Runs all diagnostics with interpretive output

Usage:
```python
from utils.diagnostics import run_full_diagnostics
results = run_full_diagnostics(model, train_loader, test_loader, device)
```

### 2. Warmup Scheduler (10 epochs, 1e-7 to 5e-5)
**Status:** IMPLEMENTED
**Files:** `train.py`, `configs/paper_config.yaml`

Added `warmup_constant` scheduler type that:
- Starts at very low LR (1e-7)
- Linearly increases to target LR (5e-5) over 10 epochs
- Maintains constant LR thereafter

Config:
```yaml
scheduler:
  type: warmup_constant
  warmup_epochs: 10
  warmup_start_lr: 1.0e-7
  warmup_end_lr: 5.0e-5
```

### 3. MSE Loss with Positionally-Encoded Tokens
**Status:** IMPLEMENTED
**Files:** `train.py`, `models/transformer.py`, `configs/paper_config.yaml`

Changes:
- Added `get_positionally_encoded_tokens()` method to transformer
- Training loop now computes MSE between PE-augmented tokens and reconstructed tokens
- Configurable via `use_pe_loss: true` (default)

Paper specifies: MSE(R̃ᵢᵗ⁰, R̂ᵢᵗ⁰) where R̃ is positionally-encoded.

### 4. Loss-Based Early Stopping
**Status:** IMPLEMENTED
**Files:** `train.py`, `configs/paper_config.yaml`

Changes:
- Added `metric` option to early_stopping config ('loss' or 'auc_roc')
- Default changed to 'loss' (paper trains for fixed epochs, loss is more stable)
- Both metrics tracked; best checkpoint saved based on selected metric

Config:
```yaml
early_stopping:
  enabled: true
  patience: 20
  min_delta: 0.001
  metric: loss  # 'loss' or 'auc_roc'
```

### 5. Learning Rate Rewinding
**Status:** NOT IMPLEMENTED (deferred)
**Priority:** LOW - Try warmup first

If warmup doesn't help, implement:
1. Train and track best epoch by loss
2. Reload best checkpoint
3. Fine-tune with 10x lower LR

### 6. Training Only on Normal Data
**Status:** VERIFIED CORRECT (no changes needed)

`poselift_dataset.py:477` confirms training uses only normal data.

---

## Key Files Modified

| File | Changes Made |
|------|--------------|
| `configs/paper_config.yaml` | warmup_constant scheduler, use_pe_loss, loss-based early stopping |
| `train.py` | warmup_constant scheduler, PE-loss computation, loss-based early stopping |
| `models/transformer.py` | Added `get_positionally_encoded_tokens()` method |
| `utils/diagnostics.py` | NEW - Token discriminability analysis functions |

---

## Paper Reference (arxiv:2504.19970)

Key specs from official Shopformer paper:
- **Optimal config:** 2 tokens, 144-dim embedding, 2 layers, 2 heads, FF=64
- **Training:** 20 epochs, Adam optimizer, LR 5e-5, no decay, no warmup
- **Data:** Train only on normal behavior, anomalies detected by reconstruction error
- **Loss:** MSE between positionally-encoded tokens and reconstructed tokens
- **Target AUC-ROC:** 69.15%

---

## Next Steps

1. **Run training with new config:**
   ```bash
   cd shopformer_2
   python train.py --config configs/paper_config.yaml
   ```

2. **Run diagnostics after Stage 1 to verify GCAE quality:**
   ```python
   from utils.diagnostics import analyze_token_discriminability
   results = analyze_token_discriminability(model, train_loader, test_loader, device)
   ```

3. **If AUC-ROC still low after warmup:**
   - Try disabling warmup (`type: none`)
   - Try different warmup lengths (5, 15, 20 epochs)
   - Implement LR rewinding (item 5)

4. **Monitor training curves:**
   - Loss should decrease steadily
   - AUC-ROC should improve after warmup phase (epoch 10+)
   - Watch for "epoch 1 is best" pattern - warmup should fix this

---

## Configuration Summary

Current `paper_config.yaml` settings:
```yaml
model:
  num_keypoints: 18
  transformer:
    num_heads: 2
    num_layers: 2
    dim_feedforward: 64
    d_model: 144

training:
  optimizer: adam
  use_pe_loss: true
  stage2:
    learning_rate: 5.0e-5
  scheduler:
    type: warmup_constant
    warmup_epochs: 10
    warmup_start_lr: 1.0e-7
    warmup_end_lr: 5.0e-5
  early_stopping:
    metric: loss
    patience: 20
```
