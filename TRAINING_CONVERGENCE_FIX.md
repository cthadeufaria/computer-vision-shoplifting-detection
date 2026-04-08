# STG-NF Early Convergence Fix (Multi-Dataset)

## Problem

Loss on epoch 3 is nearly identical to epoch 30. The model plateaus after just a few epochs and wastes the remaining training budget.

## Root Causes

1. **ActNorm data-dependent initialization** — on the very first forward pass, ActNorm layers initialize their scale/bias to whiten activations to zero-mean/unit-variance. This gives the flow most of its "easy" NLL improvement before any gradient update happens.

2. **Input is already normalized** — `normalize_pose()` applies three normalization steps (divide by resolution → subtract per-sequence mean → divide by Y-axis std). Data arrives nearly Gaussian-shaped, so ActNorm finishes the job immediately and coupling layers have little residual structure to learn.

3. **Low model capacity** — defaults `K=8, L=1, hidden_dim=0` saturate within 2-3 epochs on a large, diverse multi-dataset.

4. **LR decay compounds the plateau** — `lr = 5e-4 * 0.99^epoch` is fine numerically, but because gradients are near-zero after saturation, the optimizer stops making useful updates.

## Fixes

### 1. Increase Model Capacity (most impactful)

```bash
python train_eval.py ... --K 16 --L 2 --model_hidden_dim 128
```

- `--K 16`: more flow steps per level → richer transformations
- `--L 2`: adds a second flow level with temporal squeeze
- `--model_hidden_dim 128`: adds a hidden projection in the ST-GCN coupling network

### 2. Use Cosine LR Schedule

```bash
python train_eval.py ... --model_sched cosine --epochs 50
```

Prevents premature LR decay from freezing optimization once the model starts finding useful gradients.

### 3. Slow Down Exponential Decay

If you prefer to keep `exp_decay`, reduce the decay rate:

```bash
python train_eval.py ... --model_lr_decay 0.995
```

### 4. Lower Initial Learning Rate

Prevents ActNorm from over-fitting to the first batch's statistics:

```bash
python train_eval.py ... --model_lr 1e-4
```

### 5. Ensure a Shuffled, Representative First Batch

ActNorm initializes permanently from the first batch. Make sure the DataLoader is shuffled (`shuffle=True`) and the first batch is not dominated by a single dataset source.

## Diagnostic

Check TensorBoard (`runs/` directory) and look at the per-batch NLL loss curve:

- **Steep drop in epoch 1, then flat** → ActNorm init effect (fix: lower LR, reduce K/L slightly, ensure first batch is representative)
- **Gradual drop that plateaus early** → capacity saturation (fix: increase K, L, hidden_dim)

## Recommended Starting Config for Multi-Dataset

```bash
python train_eval.py \
  --dataset Multi \
  --K 16 \
  --L 2 \
  --model_hidden_dim 128 \
  --model_lr 1e-4 \
  --model_sched cosine \
  --epochs 50
```

## Relevant Files

| File | Lines | Role |
|------|-------|------|
| `models/STG_NF/model_pose.py` | 280-405 | STG-NF model, ActNorm is inside each FlowStep |
| `utils/data_utils.py` | 132-157 | `normalize_pose()` — triple normalization on input |
| `utils/optim_init.py` | 18-33 | LR scheduler options |
| `args.py` | 106-113 | All default hyperparameters |
| `models/training.py` | 136-140 | TensorBoard NLL loss logging |
