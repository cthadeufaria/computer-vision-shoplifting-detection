# SOTA Survey for Shoplifting Detection (Pose-First + Related VAD)
**Generated:** 2026-01-29

This report focuses on **state-of-the-art (SOTA) models you can replicate** for a shoplifting detection use case, with emphasis on **pose-based anomaly detection** (as in Shopformer) plus relevant adjacent methods. It prioritizes **privacy, compute “lightness,” and real-world deployability**, and only includes methods tied to **published papers**.

---

## 0) How to Use This Survey

- **If you want a model that aligns with the current repo (Shopformer-style):** start with **PoseLift + Shopformer + STG-NF + TSGAD + GEPC**.
- **If you want the lightest compute head:** prioritize **STG-NF** (normalizing flow) and **MPED-RNN**; both are pose-based and avoid heavy pixel inference at the model stage.
- **If you want SOTA on general human-centric VAD (not shoplifting-specific):** consider **SPARTA/PoseWatch**, **MoPRL**, and **MoCoDAD**.
- **If you need edge-focused guidance:** see Section 7 (scored shortlist) and Section 9 (pose vs RGB trade-offs).
- **If you want citations:** use `SOTA_SURVEY_SHOPLIFTING.bib` (BibTeX with DOIs).

**Important:** Most pose-based methods assume **precomputed pose sequences**. In practice, the **pose estimator** is often the **largest compute cost**. This is a deployment decision (edge vs server) more than a model choice. See Section 6.

---

## 1) Core Surveys & References (Foundational Reading)

These surveys anchor terminology, metrics, privacy, and model families.

- **Privacy-Preserving VAD Survey (P2VAD)**: taxonomy and privacy-oriented VAD approaches. Useful for GDPR/ethics framing. [S1]
- **Skeletal VAD Survey**: deep learning methods using skeleton data; privacy and pose-centric benefits. [S2]
- **Transformer VAD Survey (2025)**: transformer-based VAD methods, compute considerations, and deployment constraints. [S3]
- **Video Anomaly Detection in 10 Years (Survey & Outlook)**: broad VAD landscape and trends. [S4]

---

## 2) Shoplifting-Specific + Retail-Focused Pose Methods (Closest to Your Use Case)

### 2.1 Shopformer (Pose-Transformer, Shoplifting-Specific)
- **Paper:** “Shopformer: Transformer-Based Framework for Detecting Shoplifting via Human Pose” (arXiv 2025). [S5]
- **Modality:** Pose sequences (privacy-preserving).
- **Key idea:** Tokenize pose sequences into compact embeddings; transformer reconstructs and scores anomalies. [S5]
- **Why it matters:** First pose-sequence transformer **explicitly for shoplifting**, evaluated on PoseLift. [S5]

### 2.2 PoseLift Dataset & Benchmark (Shoplifting, Pose-Based)
- **Paper & dataset:** PoseLift (WACV 2025) + public repository. [S6][S7]
- **Benchmark results (PoseLift):**
  - **STG-NF:** AUC-ROC 67.46, AUC-PR 84.06, EER 0.39
  - **TSGAD:** AUC-ROC 63.35, AUC-PR 39.31, EER 0.41
  - **GEPC:** AUC-ROC 60.61, AUC-PR 50.38, EER 0.38
  - **Shopformer:** AUC-ROC 69.15, AUC-PR 44.49, EER 0.38
  [S7][S8]

**Takeaway:** PoseLift is the **only public, real-world shoplifting pose benchmark**; these numbers are the **most relevant** for your repo’s target. [S6][S7]

---

## 3) Pose/Skeleton-Based Video Anomaly Detection (General SOTA, Highly Relevant)

These models are **not shoplifting-specific**, but are strong baselines or upgrades for PoseLift/Shopformer comparisons.

### 3.1 STG-NF (Normalizing Flows for Human Pose AD)
- **Paper:** ICCV 2023. [S9]
- **Modality:** Pose graph sequences.
- **Compute:** Extremely lightweight head (~1K parameters) and explicitly designed to run wherever pose estimation runs. [S9]
- **Why relevant:** Strong PoseLift baseline + best trade-off for edge deployment.

### 3.2 GEPC (Graph Embedded Pose Clustering)
- **Paper:** CVPR 2020. [S10]
- **Modality:** Pose graphs; autoencoder + clustering + Dirichlet mixture. [S10]
- **Why relevant:** Classic pose-graph baseline; included in PoseLift benchmarks.

### 3.3 MPED-RNN (Skeleton Trajectory Regularity)
- **Paper:** CVPR 2019. [S11]
- **Modality:** Skeleton trajectories, decomposed into global motion + local posture, recurrent encoder-decoder. [S11]
- **Why relevant:** Strong, older baseline; simpler than transformers.

### 3.4 TSGAD (Two-Stream Graph-Improved Anomaly Detection)
- **Paper:** arXiv 2024; uses **VAE for pose + trajectory prediction**. [S12]
- **Why relevant:** PoseLift baseline and an alternative to transformer-heavy models.

### 3.5 MoPRL (Motion Prior Regularity Learner)
- **Paper:** arXiv 2021 / TCSVT 2022. [S14][S15]
- **Modality:** Pose-based; motion embedder + spatial-temporal transformer. [S15]
- **Why relevant:** Strong results on skeleton VAD; a transformer design with a motion prior.

### 3.6 MoCoDAD (Diffusion for Skeleton VAD)
- **Paper:** ICCV 2023. [S16]
- **Modality:** Skeleton; diffusion-based multimodal future prediction.
- **Why relevant:** High accuracy on standard datasets; heavier compute than flow/RNN models.

### 3.7 BiPOCO (Bi-directional Pose-Constrained Trajectory Prediction)
- **Paper:** arXiv 2022. [S17]
- **Modality:** Pose trajectory prediction with constraints; anomaly detection via prediction error. [S17]

### 3.8 TrajREC (Multitask Trajectory Reconstruction)
- **Paper:** WACV 2024; attention-based encoder-decoder for trajectory reconstruction. [S18]
- **Modality:** Skeleton trajectories.

### 3.9 SPARTA / PoseWatch (Pose Tokenization + Transformer)
- **Paper:** arXiv 2024. [S19]
- **Modality:** Pose-based transformer with spatio-temporal pose tokenization. [S19]
- **Why relevant:** Strong transformer-based pose VAD; conceptually close to Shopformer.

---

## 4) Shoplifting-Specific RGB / Hybrid Models (Non-Pose, Still Relevant)

These are **shoplifting-specific**, but typically **pixel-heavy** and less privacy-preserving.

### 4.1 CNN-BiLSTM Shoplifting (MDPI 2023)
- **Paper:** “Shoplifting Detection Using Hybrid Neural Network CNN-BiLSTM” (Applied Sciences 2023). [S20]
- **Method:** Inception V3 features + BiLSTM; dataset of 900 videos.
- **Reported performance:** ~81% accuracy (and other metrics) on their dataset. [S20]

### 4.2 Automatic Shoplifting Detection (AAAI 2020 Student Abstract)
- **Paper:** ROI optical-flow fusion network for shoplifting detection. [S21]
- **Notes:** Short abstract paper; limited detail but a useful historical reference.

### 4.3 Pre-Shoplifting Suspicious Behavior (IEEE 2024)
- **Paper:** “Detection of Pre-Shoplifting Suspicious Behavior Using Deep Learning.” [S22]
- **Notes:** Early detection (pre-theft) focus; can complement anomaly detection pipelines.

---

## 5) Applicability to Shoplifting (Evidence & Metrics)

### 5.1 What’s actually evaluated on shoplifting data?
- **PoseLift + Shopformer + STG-NF + TSGAD + GEPC** are the only **public benchmarked, pose-based shoplifting results** in open literature (WACV 2025 + Shopformer paper). [S6][S7][S8]

### 5.2 Metrics that matter for real-world deployment
- Research uses **AUC-ROC, AUC-PR, EER**, but **real deployment needs FPR/FNR at operating thresholds**, because false alerts are costly in retail. PoseLift includes EER but not detailed deployment curves. [S6][S7]
- Privacy requirements and bias considerations are a key part of practical VAD deployment. [S1][S2]

---

## 6) Lightness & Deployment Considerations

### 6.1 Two-stage pipeline cost (Pose Estimation + Anomaly Model)
- Almost all pose-based papers assume **pose sequences already available**; the pose estimator is usually the heaviest component.
- For **edge deployment**, choose a **light pose estimator** + **light anomaly head** (e.g., STG-NF). [S9]

### 6.2 Comparative lightness (qualitative)
- **Very light model head:** STG-NF (normalizing flow head, ~1K parameters). [S9]
- **Light-to-medium head:** MPED-RNN, GEPC (RNN / clustering on pose graphs). [S10][S11]
- **Medium-to-heavy:** Transformers (Shopformer, SPARTA, MoPRL) and Diffusion (MoCoDAD). [S5][S15][S16][S19]

*Note: “lightness” here refers to the anomaly model head. Actual runtime depends heavily on pose extraction and input resolution.*

---

## 7) Edge-Device Shortlist (Scored, Heuristic)

**Scoring rubric (1-5, higher is better):**  
- **Head Lightness (HL):** smaller/leaner anomaly head.  
- **Pipeline Simplicity (PS):** pose-only (5) vs RGB-heavy (1-2).  
- **Overall Edge Score:** rounded average of HL and PS.  

**Important:** This is a **heuristic** ranking based on model class and authors’ stated design goals (e.g., explicit lightweight claims). It **does not** include pose-estimator cost, which is often the dominant runtime factor. [S9]

| Model | HL | PS | Overall Edge Score | Notes |
| --- | --- | --- | --- | --- |
| **STG-NF** | 5 | 5 | **5** | Explicitly lightweight (~1K parameters) pose-only head. [S9][S23] |
| **GEPC** | 4 | 5 | **4** | Pose graph clustering; leaner than transformers. [S10][S24] |
| **MPED-RNN** | 4 | 5 | **4** | RNN-based pose modeling; simpler than transformers. [S11] |
| **TSGAD** | 3 | 5 | **4** | VAE + trajectory prediction; mid-weight. [S12] |
| **BiPOCO** | 3 | 5 | **4** | Trajectory prediction; moderate head complexity. [S26] |
| **TrajREC** | 3 | 5 | **4** | Attention encoder-decoder; mid-weight. [S27] |
| **Shopformer** | 2 | 5 | **3** | Transformer-heavy head; still pose-only. [S5][S8] |
| **SPARTA/PoseWatch** | 2 | 5 | **3** | Transformer with pose tokenization. [S19][S28] |
| **MoCoDAD** | 1 | 5 | **3** | Diffusion model; heavy compute head. [S16][S29] |
| **CNN-BiLSTM (RGB)** | 2 | 2 | **2** | Pixel-heavy pipeline and lower privacy. [S20] |

**Edge takeaways:** If you need on-device inference, **STG-NF** is the safest first replication; **GEPC** and **MPED-RNN** are next-best lightweight baselines. [S9][S10][S11][S24]

---

## 8) Replication Plan (Datasets + Checkpoints + What You Can Reuse)

**Goal:** Run models that are paper-backed, with reproducible data and evaluation.  
**Primary dataset for shoplifting:** **PoseLift** (pose sequences, frame-level labels). [S6][S7]

### 8.1 Shoplifting-First (PoseLift + Shopformer-family)

**Shopformer**  
- **Paper + code:** Official repo available. [S5][S8]  
- **Data:** PoseLift. [S6][S7]  
- **Weights:** Repo indicates a pretrained tokenizer can be used; verify exact download location in the repo. [S3]  
- **Evaluation:** Frame-level AUC-ROC/AUC-PR/EER on PoseLift (same as benchmark tables). [S7][S8]

**STG-NF**  
- **Paper + code:** ICCV paper + official repo (pretrained checkpoints for ShanghaiTech/UBnormal). [S9][S23]  
- **Data:** PoseLift requires custom pose-graph formatting; STG-NF includes scripts for pose formatting and custom dataset support. [S23]  
- **Weights:** Pretrained checkpoints exist for other datasets only; PoseLift must be trained from scratch. [S23]

**TSGAD**  
- **Paper:** arXiv 2406.15395. [S12]  
- **Data:** PoseLift (pose + trajectory inputs). [S6][S7]  
- **Weights:** No PoseLift pretrained weights noted; expect to train. [S12]

**GEPC**  
- **Paper + code:** Official repo exists. [S10][S24]  
- **Data:** PoseLift poses need to be converted to GEPC pose-graph format. [S24]  
- **Weights:** No PoseLift pretrained weights noted; expect to train. [S24]

### 8.2 General SOTA Pose VAD (Useful Upgrades, Not Shoplifting-Specific)

**SPARTA / PoseWatch**  
- **Paper + code:** Official repo exists. [S19][S28]  
- **Data:** Benchmarked on general VAD datasets; can be adapted to PoseLift. [S28]  
- **Weights:** Not clearly advertised in repo; verify in README. [S28]

**MoCoDAD (Diffusion)**  
- **Paper + code:** ICCV paper + official repo (pretrained checkpoints for Avenue/UBnormal/ShanghaiTech). [S16][S29]  
- **Data:** PoseLift adaptation requires trajectory format similar to UBnormal/HR-STC. [S29]  
- **Weights:** Pretrained available for non-shoplifting datasets. [S29]

**TrajREC**  
- **Paper + code:** WACV paper + official repo (saved weights for HR-STC and HR-Avenue). [S18][S27]  
- **Data:** Trajectory format; needs PoseLift trajectory extraction. [S27]

**BiPOCO**  
- **Paper + code:** arXiv paper + official repo (pretrained models for Avenue/ShanghaiTech). [S17][S26]  
- **Data:** Trajectory format; requires AlphaPose + tracking pipeline. [S26]

### 8.3 Non-Pose Shoplifting Papers (Pixel/Hybrid)

**CNN-BiLSTM (MDPI 2023)**  
- **Paper:** shoplifting-specific CNN-LSTM method, dataset of 900 instances. [S20]  
- **Data:** Custom dataset; not PoseLift.  
- **Weights:** Not publicly reported; expect to reimplement from paper. [S20]

### 8.4 Recommended Replication Order (Best ROI)

1. **STG-NF** (fast, lightweight, PoseLift-aligned). [S9][S23]  
2. **Shopformer** (directly aligned with repo + PoseLift). [S5][S8]  
3. **TSGAD / GEPC** (PoseLift baselines; medium complexity). [S10][S12]  
4. **SPARTA** (strong pose transformer; higher complexity). [S19][S28]  
5. **MoCoDAD / TrajREC / BiPOCO** (heavier, high accuracy on general datasets). [S16][S18][S17][S29][S27][S26]

---

## 9) Pose vs RGB (Side-by-Side Trade-off for Shoplifting)

| Criterion | Pose-Only (Shopformer, STG-NF, TSGAD, GEPC) | RGB/Hybrid (CNN-BiLSTM, optical-flow fusion) |
| --- | --- | --- |
| **Privacy / GDPR** | Strong advantage; no raw RGB stored. [S1][S2] | Weaker; raw pixels increase privacy risk. [S1] |
| **Compute head** | Often lighter (RNN/flow) but still depends on pose extraction. [S9][S11] | Heavy due to CNN feature extraction. [S20] |
| **Interpretability** | Higher (pose trajectories are human-readable). [S2] | Lower (dense pixel features). |
| **Deployment risk** | Lower bias risk; easier compliance story. [S1][S2] | Higher regulatory risk in retail. [S1] |
| **Accuracy on shoplifting** | Best public evidence comes from PoseLift benchmarks. [S7][S8] | Limited shoplifting-specific benchmarks; results are dataset-dependent. [S20] |

**Recommendation:** If privacy and edge deployment are priorities, stick to **pose-based** and invest in a strong pose estimator + lightweight anomaly head (STG-NF or GEPC). [S9][S10][S24]

---

## 10) Replication Shortlist (Recommended First Targets)

**Tier 1 (Shoplifting-aligned + PoseLift):**
- Shopformer (pose transformer) [S5]
- STG-NF (pose flow, lightweight) [S9]
- TSGAD (pose+trajectory VAE) [S12]
- GEPC (pose clustering baseline) [S10]

**Tier 2 (General SOTA pose VAD, good for upgrades):**
- SPARTA/PoseWatch (pose tokenization transformer) [S19]
- MoPRL (motion prior + ST transformer) [S15]
- MoCoDAD (diffusion for skeleton VAD) [S16]

**Tier 3 (Non-pose shoplifting baselines):**
- CNN-BiLSTM shoplifting (MDPI 2023) [S20]
- AAAI 2020 ROI optical-flow fusion [S21]
- Pre-shoplifting detection (IEEE 2024) [S22]

---

## 11) Open Gaps for Real-World Shoplifting

- **False positives/negatives at operational thresholds** are rarely reported for shoplifting-specific datasets; need calibration on PoseLift or real store data. [S6][S7]
- **Domain shift** (store layout, camera view, crowd density) remains a major risk; pose-based models help but are not immune. [S2]
- **Pose quality** (occlusion, low-res cameras) strongly impacts the anomaly model; pose estimator choice is a key practical decision.

---

## 12) Recommended DOI Reading Order (Aligned to Replication Plan)

**Phase 0 — Surveys (context + metric framing):**  
1) P2VAD survey — DOI: 10.48550/ARXIV.2411.14565  
2) Skeletal VAD survey — DOI: 10.48550/ARXIV.2301.00114  
3) Transformer VAD survey — DOI: 10.1007/s00521-025-11218-1  
4) VAD 10-year outlook — DOI: 10.1007/s00521-025-11659-8  

**Phase 1 — Shoplifting dataset + primary model:**  
5) PoseLift dataset — DOI: 10.1109/WACVW65960.2025.00125  
6) Shopformer — DOI: 10.48550/ARXIV.2504.19970  

**Phase 2 — Lightweight, edge-friendly pose baselines:**  
7) STG-NF — DOI: 10.1109/ICCV51070.2023.01246  
8) GEPC — DOI: 10.1109/CVPR42600.2020.01055  
9) MPED-RNN — DOI: 10.1109/CVPR.2019.01227  
10) TSGAD — DOI: 10.48550/ARXIV.2406.15395  

**Phase 3 — Higher-capacity pose upgrades:**  
11) SPARTA/PoseWatch — DOI: 10.48550/ARXIV.2408.15185  
12) MoCoDAD — DOI: 10.1109/ICCV51070.2023.00947  
13) MoPRL — DOI: 10.48550/ARXIV.2112.03649  
14) Regularity learning (TCSVT) — DOI: 10.1109/TCSVT.2023.3296118  
15) TrajREC — DOI: 10.1109/WACV57701.2024.00659  
16) BiPOCO — DOI: 10.48550/ARXIV.2207.02281  

**Phase 4 — Non-pose shoplifting baselines (comparison only):**  
17) CNN-BiLSTM shoplifting — DOI: 10.3390/app13148341  
18) AAAI 2020 shoplifting — DOI: 10.1609/aaai.v34i10.7169  
19) Pre-shoplifting detection — DOI: 10.1109/IIAI-AAI63651.2024.00088  

---

## 13) References (Key Sources)

[S1] Privacy-Preserving Video Anomaly Detection: A Survey (arXiv 2411.14565)
- https://arxiv.org/abs/2411.14565
- DOI: 10.48550/ARXIV.2411.14565

[S2] Skeletal Video Anomaly Detection using Deep Learning: Survey, Challenges and Future Directions (arXiv 2301.00114)
- https://arxiv.org/abs/2301.00114
- DOI: 10.48550/ARXIV.2301.00114

[S3] An overview of transformers for video anomaly detection (Neural Computing and Applications, 2025)
- https://link.springer.com/article/10.1007/s00521-025-11218-1
- DOI: 10.1007/s00521-025-11218-1

[S4] Video anomaly detection in 10 years: a survey and outlook (Neural Computing and Applications, 2025)
- https://nchr.elsevierpure.com/en/publications/video-anomaly-detection-in-10-years-a-survey-and-outlook
- DOI: 10.1007/s00521-025-11659-8

[S5] Shopformer: Transformer-Based Framework for Detecting Shoplifting via Human Pose (arXiv 2504.19970)
- https://arxiv.org/abs/2504.19970
- DOI: 10.48550/ARXIV.2504.19970

[S6] PoseLift dataset paper (WACVW 2025 Workshop)
- https://openaccess.thecvf.com/content/WACV2025W/ASTAD/html/Rashvand_Exploring_Pose-Based_Anomaly_Detection_for_Retail_Security_A_Real-World_Shoplifting_WACVW_2025_paper.html
- DOI: 10.1109/WACVW65960.2025.00125

[S7] PoseLift dataset repository (benchmark metrics)
- https://github.com/TeCSAR-UNCC/PoseLift
- DOI: N/A (dataset repo)

[S8] Shopformer repository (PoseLift benchmark table)
- https://github.com/TeCSAR-UNCC/Shopformer
- DOI: N/A (code repo)

[S9] Normalizing Flows for Human Pose Anomaly Detection (ICCV 2023)
- https://openaccess.thecvf.com/content/ICCV2023/html/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.html
- DOI: 10.1109/ICCV51070.2023.01246

[S10] Graph Embedded Pose Clustering for Anomaly Detection (CVPR 2020)
- https://openaccess.thecvf.com/content_CVPR_2020/html/Markovitz_Graph_Embedded_Pose_Clustering_for_Anomaly_Detection_CVPR_2020_paper.html
- DOI: 10.1109/CVPR42600.2020.01055

[S11] Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos (CVPR 2019)
- https://openaccess.thecvf.com/content_CVPR_2019/html/Morais_Learning_Regularity_in_Skeleton_Trajectories_for_Anomaly_Detection_in_Videos_CVPR_2019_paper.html
- DOI: 10.1109/CVPR.2019.01227

[S12] TSGAD (Two-Stream Graph-Improved Anomaly Detection) (arXiv 2406.15395)
- https://arxiv.org/abs/2406.15395
- DOI: 10.48550/ARXIV.2406.15395

[S14] MoPRL (arXiv 2112.03649) metadata via DBLP
- https://dblp.org/rec/journals/corr/abs-2112-03649
- DOI: 10.48550/ARXIV.2112.03649

[S15] Regularity Learning via Explicit Distribution Modeling for Skeletal Video Anomaly Detection (TCSVT 2024)
- https://dblp.org/rec/journals/tcsvt/RenCWYF24
- DOI: 10.1109/TCSVT.2023.3296118

[S16] MoCoDAD: Multi-modal Conditional Diffusion for Skeleton Action Anomaly Detection (ICCV 2023)
- https://openaccess.thecvf.com/content/ICCV2023/html/Flaborea_MoCoDAD_Multi-Modal_Conditional_Diffusion_for_Skeleton_Action_Anomaly_Detection_ICCV_2023_paper.html
- DOI: 10.1109/ICCV51070.2023.00947

[S17] BiPOCO (arXiv 2207.02281)
- https://arxiv.org/abs/2207.02281
- DOI: 10.48550/ARXIV.2207.02281

[S18] TrajREC: Trajectory Reconstruction for VAD (WACV 2024)
- https://openaccess.thecvf.com/content/WACV2024/html/Stergiou_TrajREC_Trajectory_Reconstruction_for_Video_Anomaly_Detection_WACV_2024_paper.html
- DOI: 10.1109/WACV57701.2024.00659

[S19] SPARTA / PoseWatch (arXiv 2408.15185)
- https://arxiv.org/abs/2408.15185
- DOI: 10.48550/ARXIV.2408.15185

[S20] Shoplifting Detection Using Hybrid Neural Network CNN-BiLSTM (MDPI, 2023)
- https://www.mdpi.com/2076-3417/13/14/8341
- DOI: 10.3390/app13148341

[S21] An Automatic Shoplifting Detection from Surveillance Videos (AAAI 2020 student abstract)
- https://dblp.org/rec/conf/aaai/GimLK0N20.html
- DOI: 10.1609/aaai.v34i10.7169

[S22] Detection of Pre Shoplifting Suspicious Behavior Using Deep Learning (IEEE IIAI-AAI 2024)
- https://colab.ws/articles/10.1109%2Fiiai-aai63651.2024.00088
- DOI: 10.1109/IIAI-AAI63651.2024.00088

[S23] STG-NF official repo (pretrained checkpoints, data format notes)
- https://github.com/orhir/STG-NF
- DOI: N/A (code repo)

[S24] GEPC official repo
- https://github.com/amirmk89/gepc
- DOI: N/A (code repo)

[S26] BiPOCO official repo (code + pretrained links)
- https://github.com/akanuasiegbu/BiPOCO
- DOI: N/A (code repo)

[S27] TrajREC official repo (code + saved weights)
- https://github.com/alexandrosstergiou/TrajREC
- DOI: N/A (code repo)

[S28] SPARTA official repo
- https://github.com/TeCSAR-UNCC/SPARTA
- DOI: N/A (code repo)

[S29] MoCoDAD official repo (pretrained checkpoints)
- https://github.com/aleflabo/MoCoDAD
- DOI: N/A (code repo)
 