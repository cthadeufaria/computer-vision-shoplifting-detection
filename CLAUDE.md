# Computer Vision Shoplifting Detection - Project Context

**Last Updated:** 2026-03-08

---

## Last Session — Resume Point

**Exported transcript:** `shoplifting-threshold-tuning.txt` (project root)

**Where we stopped (2026-03-08):**
- Ran `video_inference_pipeline.py` with `--anomaly_threshold -1.2 --display --run_sinth_samples` using the `ShanghaiTech_85_9` model
- Discovered a **domain mismatch**: the ShanghaiTech model assigns *lower* (more anomalous) scores to normal shopping footage than to shoplifting footage — the threshold of -1.2 triggered 101 false positives on the Normal clip and 0 alerts on the Shoplifting clip
- Scores output: `stg_nf_official/inference_outputs/video_online/`

**Immediate next steps:**
1. Build a labeled calibration set from your own wild retail videos (`normal/` + `shoplifting/` clips)
2. Run batch inference and use the threshold-finding script (in transcript) to find the optimal threshold via F1 or FPR target
3. Consider adaptive/per-scene thresholding (`mean - k*std` over a rolling window) — more robust than a fixed global value
4. Long term: retrain or fine-tune on retail data to fix the domain mismatch at the model level

---

## Quick Start for Claude

When working on this project:
1. The main implementation is in `shopformer_2/` directory
2. Current model achieves ~58% AUC-ROC, target is 69.15% (paper) or 80%+ (commercial)
3. This is a **pose-based** approach for **privacy-preserving** shoplifting detection
4. Key advantage: GDPR compliant (no facial recognition, no biometric data)

---

## Project Goal

Build a commercially viable, privacy-preserving shoplifting detection system using human pose sequences. The approach uses skeleton/pose data only (no RGB video storage) to detect anomalous behaviors indicative of shoplifting.

---

## Current Implementation Status

### Paper Reference: Shopformer (arxiv:2504.19970)

| Metric | Current | Target (Paper) | Commercial Viable |
|--------|---------|----------------|-------------------|
| AUC-ROC | ~58% | 69.15% | 80%+ |
| False Positive Rate | Unknown | Unknown | <10% |

### Architecture
- **Stage 1:** Graph Convolutional Autoencoder (GCAE) tokenizer
- **Stage 2:** Transformer encoder-decoder for sequence reconstruction
- **Anomaly Score:** MSE reconstruction error

### Key Config (Paper-Aligned)
```yaml
tokens: 2
embedding_dim: 144
num_layers: 2
num_heads: 2
ff_dim: 64
epochs: 20
lr: 5e-5
optimizer: Adam
```

---

## Research Survey: Shoplifting Detection (2023-2025)

### Key Surveys

| Survey | Link | Focus |
|--------|------|-------|
| Privacy-Preserving VAD | https://arxiv.org/abs/2411.14565 | Privacy techniques for video anomaly detection |
| Skeletal VAD Survey | https://arxiv.org/abs/2301.00114 | Skeleton-based anomaly detection methods |
| Transformers for VAD | https://link.springer.com/article/10.1007/s00521-025-11218-1 | Transformer architectures for anomaly detection |

### SOTA Pose-Based Methods

| Method | Paper | Code | Performance | Approach |
|--------|-------|------|-------------|----------|
| **STG-NF** | https://arxiv.org/abs/2210.07355 | https://github.com/orhir/STG-NF | 85.9% AUC (ShanghaiTech), 67.46% (PoseLift) | Normalizing flows on pose graphs |
| **MoCoDAD** | https://arxiv.org/abs/2307.07152 | https://github.com/aleflabo/MoCoDAD | SOTA on UBnormal | Diffusion for multimodal futures |
| **GEPC** | https://arxiv.org/abs/2001.05280 | https://github.com/amirmk89/gepc | Baseline | Graph embedded pose clustering |
| **HD-GCN** | https://arxiv.org/abs/2208.10741 | https://github.com/Jho-Yonsei/HD-GCN | 93.0% NTU-60 | Hierarchical decomposed GCN |
| **Shopformer** | https://arxiv.org/abs/2504.19970 | https://github.com/TeCSAR-UNCC/Shopformer | 69.15% (PoseLift) | Transformer autoencoder |

### Recommended Next Architecture to Try

**Priority 1: STG-NF (Normalizing Flows)**
- Replace transformer decoder with normalizing flow
- Anomaly score = negative log-likelihood instead of MSE
- Only ~1K parameters, very lightweight
- Already achieves 67.46% on PoseLift

**Priority 2: Hybrid Approach**
- Keep current GCAE tokenizer
- Add parallel flow head alongside reconstruction
- Combine scores: `final = α * recon_error + β * neg_log_likelihood`

---

## Commercial Landscape

### Major Players

| Company | Technology | Accuracy Claimed | GDPR Status | Pricing |
|---------|------------|------------------|-------------|---------|
| **Veesion** | Gesture recognition (implicit pose) | "85-100%" | **Non-compliant (CNIL 2024)** | ~$200/month |
| **Vaak** | Pose + behavior analysis | 81% | Compliant | Unknown |
| **Lexius** | Movement-only behavioral | High | Compliant | Unknown |
| **Everseen** | Video analytics | "High" | Unknown | Enterprise |
| **Facewatch** | Facial recognition | "98%" | **Under investigation** | Unknown |

### Commercial Viability Thresholds

| Level | AUC-ROC | FPR | Status |
|-------|---------|-----|--------|
| Research | 60-70% | Any | Academic only |
| **Minimum Viable** | 80-85% | <20% | Early pilots |
| Competitive | 90%+ | <10% | Production ready |
| Premium | 95%+ | <5% | Enterprise tier |

### Your Competitive Advantage

**Privacy-First Approach:**
- Veesion ruled GDPR non-compliant by French CNIL (2024)
- Facewatch under ICO investigation
- Rite Aid banned from facial recognition by FTC (5 years)
- **Your pose-only approach is inherently GDPR compliant**

---

## Key Metrics Explained

### AUC-ROC (Area Under ROC Curve)
- Measures discrimination ability across all thresholds
- 50% = random guessing, 100% = perfect
- **Standard metric for academic benchmarks**

### False Positive Rate (FPR)
- FPR = FP / (FP + TN)
- **Critical for deployment** - high FPR causes alert fatigue
- Industry target: <5-10%
- Current systems often have 90% FP rates

### Why FPR Matters More Than AUC for Deployment
- Legal settlements for false accusations: $50K - $4.4M
- Alert fatigue: 65% of security staff admit ignoring alerts
- Customer experience: 25% of falsely flagged customers switch to competitors

---

## Benchmark Datasets

| Dataset | Size | Anomaly Types | Primary Metric | Link |
|---------|------|---------------|----------------|------|
| **PoseLift** | 153 clips | Shoplifting | AUC-ROC | https://github.com/TeCSAR-UNCC/PoseLift |
| **UCF-Crime** | 1,900 videos | 13 types (incl. shoplifting) | Frame AUC | Public |
| **ShanghaiTech** | 437 videos | Campus anomalies | Frame AUC | Public |
| **XD-Violence** | 4,754 videos | Violence | AP | Public |

---

## Commercial Reality Check

### Academic vs. Commercial Claims

| Source | Claimed | Independent Verification |
|--------|---------|--------------------------|
| Academic (PoseLift) | 67% AUC | Peer-reviewed |
| Veesion | "85-100%" | **None** |
| Facewatch | "98%" | **None** - under investigation |
| Walmart/Everseen | "High" | Employees call it "NeverSeen" |

### High-Profile Failures

| Company | Failure | Consequence |
|---------|---------|-------------|
| **Walmart/Everseen** | $500M investment, still $3B theft | Employees leaked footage of failures |
| **Rite Aid** | Thousands of false positives | FTC banned facial recognition 5 years |
| **Tesco AI** | Wrongly flagged customers | Customer backlash, social media mockery |

---

## ROI and Market Opportunity

### Market Size
- AI Theft Deterrence: $2.43B (2024) → $7.82B (2032)
- US Retail Shrink: $121.6B (2024)
- Only 2% of shoplifters currently caught

### Pricing Models
- Per camera: $500-2,500 hardware + $99-200/month subscription
- Per store: $200-500/month for small retail
- Enterprise: $350K-$1.2M deployment + $70-240K annual

### ROI Benchmarks
- 55% of retailers report ROI >10%
- 21% achieve returns >30%
- Shrink reduction: 15-60% typical
- Payback period: 12-18 months

---

## Regulatory Considerations

### EU AI Act (2024-2026)
- Real-time biometric identification: **Prohibited**
- Facial recognition databases: **Prohibited**
- Behavior-based AI: **High-risk** (requires compliance)
- Pose-only detection: **Lower risk category**

### GDPR
- Biometric data = special category requiring explicit consent
- Pose/skeleton data without identity = generally compliant
- Edge processing (no cloud storage) = preferred

### US Regulations
- No federal CCTV laws
- Illinois BIPA: Strictest state law
- 21 states have consumer privacy laws

---

## Next Steps

### Priority 1: Close AUC Gap (58% → 69%)
- [ ] Experiment with warmup configurations
- [ ] Run token discriminability diagnostics
- [ ] Compare with STG-NF baseline on PoseLift

### Priority 2: Improve Beyond Paper (69% → 80%+)
- [ ] Integrate normalizing flows (STG-NF approach)
- [ ] Add dual attention mechanisms (DA-Flow)
- [ ] Try hybrid scoring (reconstruction + density)

### Priority 3: Reduce False Positives
- [ ] Implement confidence thresholding
- [ ] Add POS integration for transaction correlation
- [ ] Develop human-in-the-loop review system

### Priority 4: Commercial Preparation
- [ ] Benchmark FPR alongside AUC
- [ ] Test on diverse store environments
- [ ] Document GDPR compliance advantages

---

## Key Files

| File | Purpose |
|------|---------|
| `shopformer_2/configs/paper_config.yaml` | Paper-aligned configuration |
| `shopformer_2/train.py` | Training loop |
| `shopformer_2/models/transformer.py` | Transformer architecture |
| `shopformer_2/CLAUDE.md` | Implementation-specific notes |

---

## Commands

```bash
# Train with paper config
cd shopformer_2
python train.py --config configs/paper_config.yaml

# Clone STG-NF for comparison
git clone https://github.com/orhir/STG-NF

# Clone MoCoDAD for comparison
git clone https://github.com/aleflabo/MoCoDAD
```

---

## Sources

### Academic Papers
- PoseLift: https://arxiv.org/abs/2501.06591
- Shopformer: https://arxiv.org/abs/2504.19970
- STG-NF: https://arxiv.org/abs/2210.07355
- MoCoDAD: https://arxiv.org/abs/2307.07152

### Commercial Analysis
- Veesion: https://veesion.io
- CNIL Ruling: https://gdprhub.eu/index.php?title=CE_-_495153
- FTC Rite Aid: https://www.ftc.gov/news-events/news/press-releases/2023/12/rite-aid-banned-using-ai-facial-recognition

### Industry Reports
- NRF Retail Security Survey: https://nrf.com/research/national-retail-security-survey-2023
- AI Theft Deterrence Market: https://www.newstrail.com/ai-driven-retail-theft-deterrence-market/
