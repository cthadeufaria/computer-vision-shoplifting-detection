# Shopformer_2 Project Context
**Last Updated:** 2026-01-02

---

## Recent Research & Related Work

### Comprehensive Surveys on Video Anomaly Detection

| Paper | Year | Key Focus |
|-------|------|-----------|
| [Video Anomaly Detection in 10 Years: A Survey and Outlook](https://arxiv.org/html/2405.19387v1) | 2024 | Weakly supervised, self-supervised, unsupervised VAD approaches |
| [Deep Learning-Based Anomaly Detection in Video Surveillance: A Survey](https://www.mdpi.com/1424-8220/23/11/5024) | 2023 | Abnormal human behavior recognition in surveillance |
| [A Comprehensive Review on Deep Learning-Based Methods for Video Anomaly Detection](https://www.sciencedirect.com/science/article/abs/pii/S0262885620302109) | 2020 | IVSS for crime detection |

### Shoplifting-Specific Papers (2024-2025)

#### Transformer-Based
- **[Shopformer: Transformer-Based Framework for Detecting Shoplifting via Human Pose](https://arxiv.org/html/2504.19970)** - First pose-sequence-based transformer for shoplifting detection. Custom tokenization for pose sequences. Code: https://github.com/TeCSAR-UNCC/Shopformer

#### CNN-LSTM Hybrids
- **[Shoplifting Detection from Customer Behavior Using Deep Learning](https://link.springer.com/article/10.1007/s11760-025-04543-4)** (2025) - Time Distributed CNN-LSTM + YOLOv4 + Siamese Networks. Integrates person detection, activity recognition, product detection, re-identification.
- **[Shoplifting Detection Using Hybrid Neural Network CNN-BiLSTM](https://www.mdpi.com/2076-3417/13/14/8341)** (2023) - Benchmark dataset: 900 instances, 5 shoplifting methods.

#### Pre-Shoplifting Detection
- **[Detection of Pre-Shoplifting Suspicious Behavior Using Deep Learning](https://ieeexplore.ieee.org/document/10707900/)** (IEEE, 2024) - CNN + BiLSTM for early detection of behaviors preceding shoplifting.

#### Pose-Based / Privacy-Preserving
- **[PoseLift: Exploring Pose-Based Anomaly Detection for Retail Security](https://arxiv.org/html/2501.06591v1)** (2025) - Privacy-preserving dataset using pose-based anomaly detection (STG-NF, GEPC, TSGAD). Dataset: https://github.com/TeCSAR-UNCC/PoseLift

#### Temporal Feature Approaches
- **[Suspicious Behavior Detection with Temporal Feature Extraction and Time-Series Classification](https://www.mdpi.com/1424-8220/23/13/5811)** (2023) - YOLOv5 + DeepSort tracking, time-series classification. 92% F1 score.

### Key Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| UCF-Crime | ~128 hours video, shoplifting among other crimes | Public |
| PoseLift | Real-world shoplifting with pose data (privacy-preserving) | https://github.com/TeCSAR-UNCC/PoseLift |
| CNN-BiLSTM Benchmark | 900 instances, 5 shoplifting methods | MDPI paper |

### Research Trends to Explore

1. **Privacy-preserving approaches** - Pose estimation instead of raw video (addresses FRT policy concerns)
2. **Transformer architectures** - Outperforming CNN-LSTM hybrids on large datasets
3. **Pre-shoplifting detection** - Earlier intervention by detecting suspicious behavior before theft
4. **Unsupervised/weakly-supervised learning** - Address labeled data scarcity
5. **Vision Transformers (ViT)** - Self-attention for spatiotemporal relationships across video sequences

---

## Current Implementation Status

### Paper Reference (arxiv:2504.19970)

Key specs from official Shopformer paper:
- **Optimal config:** 2 tokens, 144-dim embedding, 2 layers, 2 heads, FF=64
- **Training:** 20 epochs, Adam optimizer, LR 5e-5, no decay, no warmup
- **Data:** Train only on normal behavior, anomalies detected by reconstruction error
- **Loss:** MSE between positionally-encoded tokens and reconstructed tokens
- **Target AUC-ROC:** 69.15%

### Current Results
- AUC-ROC: ~58% (vs paper's 69.15%)
- All paper-aligned configurations implemented
- Warmup scheduler, PE-loss, loss-based early stopping in place

---

## Future Work & Next Steps

### Priority 1: Close the AUC-ROC Gap
- [ ] Experiment with different warmup lengths (5, 15, 20 epochs)
- [ ] Try disabling warmup entirely (`type: none`) to match paper exactly
- [ ] Implement LR rewinding: train, find best epoch, reload, fine-tune with 10x lower LR
- [ ] Run token discriminability diagnostics to verify GCAE quality

### Priority 2: Architecture Experiments
- [ ] Compare with STG-NF, GEPC, TSGAD models from PoseLift paper
- [ ] Explore Vision Transformer (ViT) adaptations for pose sequences
- [ ] Test pre-shoplifting detection approach (CNN + BiLSTM for early detection)

### Priority 3: Data & Evaluation
- [ ] Evaluate on PoseLift dataset for comparison with other methods
- [ ] Consider augmentation strategies for pose data
- [ ] Benchmark against hybrid CNN-LSTM approaches from recent papers

---

## Key Files

| File | Purpose |
|------|---------|
| `configs/paper_config.yaml` | Paper-aligned configuration |
| `train.py` | Training loop with warmup scheduler, PE-loss |
| `models/transformer.py` | Transformer with `get_positionally_encoded_tokens()` |
| `utils/diagnostics.py` | Token discriminability analysis functions |

---

## Commands

```bash
# Train with paper config
python train.py --config configs/paper_config.yaml

# Run diagnostics after training
python -c "from utils.diagnostics import run_full_diagnostics; ..."
```
