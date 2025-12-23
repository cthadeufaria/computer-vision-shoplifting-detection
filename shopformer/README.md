# Shopformer: Pose-Based Shoplifting Detection

Implementation of the Shopformer architecture for detecting shoplifting behavior
using human pose sequences.

## Paper Reference

> Rashvand et al., "Shopformer: Transformer-Based Framework for Detecting
> Shoplifting via Human Pose", CVPR 2025
>
> - Paper: [arXiv:2504.19970](https://arxiv.org/abs/2504.19970)
> - Official Repo: [TeCSAR-UNCC/Shopformer](https://github.com/TeCSAR-UNCC/Shopformer)

## Architecture Overview

Shopformer uses a two-stage training approach:

1. **Stage 1 - GCAE Training**: Train a Graph Convolutional Autoencoder to learn
   meaningful pose embeddings through reconstruction.

2. **Stage 2 - Transformer Training**: Freeze the GCAE encoder and train a
   transformer encoder-decoder to reconstruct token sequences. High reconstruction
   error indicates anomalous (shoplifting) behavior.

```
Pose Sequence → [GCAE Encoder] → Tokens → [Transformer] → Reconstructed Tokens
                                              ↓
                                    Reconstruction Error = Anomaly Score
```

## Installation

```bash
cd shopformer
pip install -r requirements.txt
```

## Dataset

This implementation uses the [PoseLift](https://github.com/TeCSAR-UNCC/PoseLift)
dataset - the only available real-world, pose-based benchmark for shoplifting detection.

Download the dataset and place it in `data/PoseLift/`. The expected structure:
```
data/PoseLift/
├── Pickle_files/
│   ├── Train/
│   │   └── <camera>_<video>.pkl
│   ├── Test/
│   │   └── <camera>_<video>.pkl
│   └── GT/
│       └── <camera>_<video>.npy
└── ...
```

## Training

### With PoseLift Dataset
```bash
python train.py --data_dir ./data/PoseLift
```

### With Synthetic Data (for testing)
```bash
python train.py --use_synthetic
```

### Full Training Options
```bash
python train.py \
    --data_dir ./data/PoseLift \
    --output_dir ./checkpoints \
    --seq_len 12 \
    --num_tokens 2 \
    --stage1_epochs 10 \
    --stage2_epochs 20 \
    --batch_size 32 \
    --lr 5e-5
```

## Inference

```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/PoseLift
```

## Model Configuration

Default configuration (from paper):

| Parameter | Value |
|-----------|-------|
| Sequence Length | 12 frames |
| Number of Tokens | 2 |
| Token Dimension | 144 (8 channels × 18 keypoints) |
| Transformer Heads | 2 |
| Transformer Layers | 2 |
| Feed-forward Dim | 64 |
| Dropout | 0.1 |
| Learning Rate | 5×10⁻⁵ |
| Optimizer | Adam |

## Usage in Code

```python
from shopformer import Shopformer

# Create model
model = Shopformer(
    in_channels=2,
    num_keypoints=17,
    seq_len=12,
    num_tokens=2,
    transformer_heads=2,
    transformer_layers=2
)

# Input: (batch, 2, seq_len, num_keypoints)
poses = torch.randn(16, 2, 12, 17)

# Get anomaly scores
output = model(poses)
scores = output['normality_score']  # Higher = more anomalous

# Binary prediction
predictions = model.predict(poses, threshold=0.5)
```

## Performance

The original Shopformer achieves **69.15% AUC-ROC** on the PoseLift dataset,
outperforming other pose-based anomaly detection methods.

## License

Apache-2.0 (following the original implementation)
