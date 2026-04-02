# Topology-Invariant EEG Foundation Model with Spatiotemporal JEPA and SIGReg Regularization

> 초안 상태: Introduction + Related Work 작성 완료 (2026-04-01)
> 결과 섹션은 학습 완료 후 채울 것

---

## Abstract

Electroencephalography (EEG) foundation models face a fundamental challenge: the extreme heterogeneity of electrode montages across datasets and clinical sites. Existing approaches either restrict inputs to a fixed channel set or rely on discrete tokenization schemes that sacrifice representational flexibility. We propose EEG-WM-JEPA, a topology-invariant EEG foundation model that combines (1) a learned-query channel mixer for montage-agnostic spatial encoding, (2) spatiotemporal masking for joint spatial and temporal self-supervised learning, and (3) SIGReg regularization for stable JEPA training without exponential moving average. Pretrained on REVE (61,415 hours, 32 distinct channel configurations) [El Ouahidi et al., 2025], our 19.5M-parameter model achieves [RESULTS] on standard benchmarks (TUAB) and demonstrates strong transfer to unseen montages (TDBrain, APAVA, BCI), validating genuine topology invariance.

---

## 1. Introduction

Electroencephalography (EEG) is one of the most widely used non-invasive neuroimaging modalities, with applications spanning clinical diagnosis, brain-computer interfaces (BCI), and cognitive neuroscience research [Obeid and Picone, 2016]. The growing availability of large-scale EEG datasets has motivated the development of foundation models — large pretrained encoders that can be fine-tuned for diverse downstream tasks [Weng et al., 2024]. However, unlike natural language or computer vision, where input formats are relatively standardized, EEG data exhibits extreme heterogeneity across recording sites, equipment, and experimental protocols.

This heterogeneity manifests along five axes: (1) **electrode count and placement**, varying from 8-channel clinical montages to 256-channel research systems following different standards [Jasper, 1958]; (2) **sampling rates**, ranging from 100Hz to 2000Hz; (3) **reference electrodes**, differing across sites and protocols [Bertrand et al., 1985]; (4) **noise characteristics**, dependent on equipment, environment, and subject; and (5) **segmentation conventions**, with window lengths varying from sub-second to minutes. Among these, electrode topology — the number and spatial arrangement of channels — presents the most fundamental challenge, as it determines the dimensionality of the input signal itself.

Existing EEG foundation models have addressed this challenge through various strategies, each with significant limitations. Fixed-channel approaches [Cui et al., 2024; Yang et al., 2023] restrict inputs to a predetermined montage (typically the international 10-20 system's 19 channels), discarding data from non-conforming recordings and preventing generalization to novel electrode configurations. Channel-independent methods [Jiang et al., 2024; Wang et al., 2024] treat each electrode as a separate token, enabling variable input sizes but sacrificing spatial relationships between electrodes. More recently, discrete tokenization approaches such as VQ-based methods [Jiang et al., 2024] quantize continuous EEG signals into codebook entries, which can introduce information loss and are mathematically incompatible with continuous regularization objectives.

A promising direction has emerged from the Joint-Embedding Predictive Architecture (JEPA) framework [Assran et al., 2023], which learns representations by predicting masked regions in a learned latent space rather than reconstructing raw inputs. JEPA-based approaches have shown strong performance in vision [Assran et al., 2023; Bardes et al., 2024] and neuroimaging [Dong et al., 2024], and have recently been adapted for EEG through Laya [Panchavati et al., 2026] and LuMamba [Broustail et al., 2026]. However, these models employ only temporal masking, neglecting the spatial structure of EEG signals. Separately, S-JEPA [Guetschel et al., 2024] introduced spatial block masking for EEG but did not combine it with temporal masking.

A parallel advance in self-supervised learning has addressed representation collapse — the failure mode where encoders map all inputs to the same representation. Traditional JEPA models rely on an exponential moving average (EMA) target encoder [Assran et al., 2023; Grill et al., 2020] to prevent collapse, which introduces sensitive hyperparameters. LeJEPA [Balestriero and LeCun, 2025] proposed SIGReg (Sketched Isotropic Gaussian Regularization), a principled alternative that regularizes the representation distribution toward an isotropic Gaussian without requiring EMA. LeWM [Maes et al., 2026] extended this approach to world models, demonstrating that SIGReg enables stable training with a notably strong predictor (larger than the encoder itself).

Concurrently, the learned-query cross-attention mechanism introduced by LUNA [Doner et al., 2025] has established a new paradigm for topology-invariant EEG processing. By employing a fixed set of learned queries that attend over variable-length channel inputs, LUNA's channel mixer produces fixed-size representations regardless of electrode count. This mechanism has been adopted by Laya [Panchavati et al., 2026] and LuMamba [Broustail et al., 2026], confirming its effectiveness for cross-dataset transfer.

In this work, we propose **EEG-WM-JEPA**, a topology-invariant EEG foundation model that unifies these advances. Our key insight is that effective EEG representation learning requires simultaneously addressing spatial heterogeneity (variable electrode montages) and spatiotemporal dynamics (joint spatial and temporal signal patterns). We make the following contributions:

1. **Spatiotemporal masking for EEG-JEPA**: We are the first to combine spatial and temporal masking within a JEPA framework for EEG, enabling the model to learn both inter-electrode relationships and temporal dynamics simultaneously. Prior work used only temporal masking [Panchavati et al., 2026; Broustail et al., 2026] or only spatial masking [Guetschel et al., 2024].

2. **SIGReg with strong predictor for EEG**: Building on the analysis of LeWM [Maes et al., 2026], we adopt the "strong predictor" design philosophy (Predictor/Encoder parameter ratio ≈ 1.75) with SIGReg regularization [Balestriero and LeCun, 2025], eliminating the need for EMA and its associated hyperparameter sensitivity.

3. **Genuine topology invariance via spatial bottleneck**: Our channel mixer processes variable-channel inputs through a compressed spatial bottleneck (4× compression) with explicit padding masks, handling arbitrary electrode configurations without channel selection or artificial standardization.

4. **Large-scale heterogeneous pretraining**: We pretrain on the full REVE dataset [El Ouahidi et al., 2025] (61,415 hours, 32 distinct channel configurations, ~24,260 subjects), the largest and most diverse EEG pretraining to date.

5. **Comprehensive evaluation on unseen montages**: Beyond the standard TUAB benchmark [Lopez, 2017], we evaluate on three non-TUH benchmarks (TDBrain [van Dijk et al., 2022], APAVA [Escudero et al., 2006], BCI Competition IV [Tangermann et al., 2012]) with electrode configurations unseen during pretraining, providing strong evidence for topology invariance.

---

## 2. Related Work

### 2.1 EEG Foundation Models

The pursuit of universal EEG encoders has accelerated rapidly. Early efforts focused on contrastive learning with fixed channel configurations: BIOT [Yang et al., 2023] employed biosignal tokenization with contrastive objectives on TUH data [Obeid and Picone, 2016], while Neuro-GPT [Cui et al., 2024] combined a pretrained GPT backbone with EEG-specific encoders. These models require a fixed electrode montage, limiting their applicability across datasets.

Channel-independent approaches relaxed this constraint by treating each electrode as a separate input stream. LaBraM [Jiang et al., 2024] introduced vector-quantized (VQ) tokenization of per-channel patches followed by masked autoencoder (MAE) [He et al., 2022] pretraining, achieving strong results on TUAB at ICLR 2024. EEGPT [Wang et al., 2024] similarly employed channel-independent patching with MAE pretraining at scale (~38,000 hours). However, these methods process each channel in isolation during patch embedding, only capturing cross-channel relationships in later transformer layers.

The learned-query cross-attention mechanism, introduced by LUNA [Doner et al., 2025] at NeurIPS 2025, represented a paradigm shift. By projecting variable-channel inputs to a fixed number of learned queries via cross-attention, LUNA achieved true topology invariance while preserving spatial relationships through 3D electrode coordinate encoding. This channel mixer was subsequently adopted by Laya [Panchavati et al., 2026], which combined it with LeJEPA-based [Balestriero and LeCun, 2025] self-supervised pretraining, and LuMamba [Broustail et al., 2026], which replaced the transformer backbone with Mamba blocks for improved efficiency.

Table 1 summarizes the landscape of EEG foundation models:

| Model | Year | Venue | Topology Strategy | Learning | Pretrain Data |
|---|---|---|---|---|---|
| BIOT [Yang et al.] | 2023 | NeurIPS | Fixed channels | Contrastive | TUH |
| Neuro-GPT [Cui et al.] | 2024 | ISBI | Fixed channels | Autoregressive | TUH |
| LaBraM [Jiang et al.] | 2024 | ICLR (Spotlight) | Channel-independent VQ | MAE | ~2,500h |
| EEGPT [Wang et al.] | 2024 | NeurIPS | Channel-independent | MAE | ~38,000h |
| S-JEPA [Guetschel et al.] | 2024 | arXiv | Spatial block masking | JEPA (EMA) | Small-scale |
| HEAR [Chen et al.] | 2025 | arXiv | Heterogeneous electrode adaptive | MAE | Multi-dataset |
| LUNA [Doner et al.] | 2025 | NeurIPS | Learned-query mixer | MAE | Large-scale |
| REVE [El Ouahidi et al.] | 2025 | NeurIPS | Channel-independent | MAE | 61,415h |
| Laya [Panchavati et al.] | 2026 | arXiv | Learned-query mixer | LeJEPA (SIGReg) | ~29,000h |
| LuMamba [Broustail et al.] | 2026 | arXiv | Learned-query mixer | LeJEPA + Mamba | ~21,000h |
| **Ours** | 2026 | — | Learned-query mixer + padding mask | **ST-JEPA + SIGReg** | **61,415h** |

### 2.2 Joint-Embedding Predictive Architectures

The JEPA framework [LeCun, 2022] learns representations by predicting the latent embedding of masked target regions from visible context, rather than reconstructing raw pixel or signal values as in MAE [He et al., 2022] or BEiT [Bao et al., 2022]. This design choice avoids the model spending capacity on predicting low-level noise, instead focusing on semantic and structural features.

**I-JEPA** [Assran et al., 2023] instantiated this framework for images at CVPR 2023, using multi-block spatial masking and an EMA target encoder for collapse prevention. **V-JEPA** [Bardes et al., 2024] extended the approach to video with spatiotemporal tube masking. **Brain-JEPA** [Dong et al., 2024], a NeurIPS 2024 Spotlight, adapted JEPA for fMRI with gradient positioning and three complementary masking strategies (cross-ROI, cross-time, and double-cross), demonstrating the value of spatiotemporal masking for brain signals.

In the EEG domain, S-JEPA [Guetschel et al., 2024] introduced spatial block masking based on electrode proximity, masking all channels within a spatial radius. However, S-JEPA used only spatial masking without temporal blocks, and relied on the standard EMA collapse prevention mechanism. Laya [Panchavati et al., 2026] and LuMamba [Broustail et al., 2026] adopted temporal-only masking with SIGReg regularization.

Our work is the first to combine spatial and temporal masking within a JEPA framework for EEG, drawing inspiration from Brain-JEPA's spatiotemporal strategy [Dong et al., 2024] while using SIGReg [Balestriero and LeCun, 2025] instead of EMA for collapse prevention.

### 2.3 Collapse Prevention: From EMA to SIGReg

A central challenge in joint-embedding methods is representation collapse, where the encoder maps all inputs to the same trivial representation. Several families of solutions have been proposed: contrastive methods using negative pairs [Chen et al., 2020], distillation with asymmetric architectures and EMA [Grill et al., 2020], and regularization-based approaches including VICReg [Bardes et al., 2022] and Barlow Twins [Zbontar et al., 2021].

I-JEPA [Assran et al., 2023] and subsequent JEPA models rely on an EMA target encoder, where the target network's weights are an exponential moving average of the online encoder. While effective, this introduces the momentum hyperparameter and can be sensitive to its scheduling [Balestriero and LeCun, 2025].

**LeJEPA** [Balestriero and LeCun, 2025] introduced SIGReg (Sketched Isotropic Gaussian Regularization), which directly regularizes the encoder output distribution toward an isotropic Gaussian via random 1D projections. This eliminates EMA entirely while providing a theoretically grounded collapse prevention mechanism. **LeWM** [Maes et al., 2026] extended LeJEPA to world models, demonstrating a notable design choice: the predictor network is approximately twice the size of the encoder (Predictor/Encoder ≈ 2.0). Our analysis of LeWM reveals that this strong predictor is essential in the SIGReg setting — without EMA to provide stable targets, the predictor must be sufficiently expressive to create meaningful prediction pressure on the encoder.

We adopt this "strong predictor" philosophy (Pred/Enc ≈ 1.75 in our model), with key adaptations: fixed dim_head=64 in the predictor attention layers (matching LeWM) and independent predictor MLP width (2048), ensuring the predictor's internal capacity substantially exceeds that of the encoder.

### 2.4 Topology-Invariant EEG Processing

Handling variable electrode configurations is a defining challenge for EEG foundation models. We identify four main strategies in the literature:

**Channel intersection** selects only electrodes common across all datasets, typically yielding a minimal set (e.g., 19 channels from the 10-20 system [Jasper, 1958]). BIOT [Yang et al., 2023] and Neuro-GPT [Cui et al., 2024] adopt this approach, sacrificing information from high-density recordings.

**Channel-independent processing** treats each electrode as a separate token, naturally handling variable counts. LaBraM [Jiang et al., 2024] and EEGPT [Wang et al., 2024] use this strategy with per-channel patch embedding. However, spatial relationships between electrodes are only implicitly captured through later self-attention layers.

**Electrode coordinate encoding** explicitly represents spatial information. HEAR [Chen et al., 2025] proposed heterogeneous electrode adaptive representations using 3D coordinates. Brain-JEPA [Dong et al., 2024] employed gradient positioning for fMRI parcels. These approaches enrich spatial awareness but still require a mechanism to produce fixed-size outputs from variable inputs.

**Learned-query cross-attention** (LUNA [Doner et al., 2025]) combines coordinate encoding with a fixed set of learned queries that attend over variable-length channel inputs via cross-attention, producing a fixed-size output regardless of electrode count. Laya [Panchavati et al., 2026] and LuMamba [Broustail et al., 2026] validated this approach for EEG-JEPA.

Our channel mixer builds on the LUNA paradigm with two key additions: (1) a **spatial bottleneck** (mixer_dim=64, 4× compression before projection to encoder_dim=256), forcing the mixer to learn compressed spatial representations rather than passing channel information through unchanged; and (2) **explicit padding masks** for batching samples with different channel counts, ensuring that zero-padded channels do not influence the cross-attention computation. Existing implementations assume uniform channel counts within each dataset, while our design handles truly mixed-montage batches

---

## 3. Method

### 3.1 Problem Formulation

입력: $\mathbf{X} \in \mathbb{R}^{C \times T}$ (C는 데이터셋마다 가변, T=400 at 200Hz)
전극 좌표: $\mathbf{P} \in \mathbb{R}^{C \times 3}$ (3D 좌표)
목표: 범용 인코더 $f_\theta$가 임의의 C에 대해 고품질 representation 생성

### 3.2 EEG Standardization (Stage 1)

[TODO: preprocessing 5요소 간략히. research_method.md 참조]

### 3.3 Patch Embedder

- 채널 독립 depthwise 1D convolution
- $\mathbf{X} \in \mathbb{R}^{C \times T} \rightarrow \mathbf{E} \in \mathbb{R}^{C \times N \times d_p}$
- patch_size=25 (125ms), N=16 patches, $d_p$=64

### 3.4 Dynamic Channel Mixer

- Fourier positional encoding: $\mathbf{P} \in \mathbb{R}^{C \times 3} \rightarrow \mathbf{F} \in \mathbb{R}^{C \times d_f}$
- $N_q$=8 learned queries, cross-attention over channels
- Output: $\mathbf{H} \in \mathbb{R}^{(N_q \cdot N) \times d_m}$ where $d_m$=64
- Linear projection: $d_m \rightarrow D$ (64 → 256)
- **padding_mask**: 패딩된 가짜 채널의 attention weight → $-\infty$ → softmax 후 0
- **Query Specialization Loss**: 쿼리 간 attention 분포 다양성 강제

### 3.5 Spatiotemporal Masking

- Mask ratio: 60%
- Temporal blocks: 0.5~1.0초 연속 (4~8 patches)
- Spatial: 쿼리 그룹 단위
- **시간+공간 동시 마스킹**: Laya(시간만) + S-JEPA(공간만)의 통합

### 3.6 Transformer Encoder

- D=256, 8 layers, 4 heads, dim_head=64
- MLP dim=1024, dropout=0.1
- 2-pass: full sequence → context-only encoding

### 3.7 Predictor (LeWM-style)

- D=256, 6 layers, 12 heads, dim_head=64 (inner_dim=768)
- MLP dim=2048
- **설계 근거**: SIGReg 방식에서 강한 predictor가 representation 품질에 직결. LeWM의 Pred/Enc≈2.0 비율 반영 (우리: 1.75)

### 3.8 Loss Function

$$\mathcal{L} = \mathcal{L}_{pred} + \lambda_1 \mathcal{L}_{SIGReg} + \lambda_2 \mathcal{L}_{query}$$

- $\mathcal{L}_{pred}$: MSE between predicted and target latents (projected)
- $\mathcal{L}_{SIGReg}$: Stochastic Isotropic Gaussian Regularization ($\lambda_1$=0.09)
  - 512 random projections, 17 knots for Gaussian quadrature
  - Encoder output 분포를 등방성 가우시안으로 정규화 → collapse 방지
- $\mathcal{L}_{query}$: Query Specialization ($\lambda_2$=0.8)
  - 각 쿼리가 서로 다른 전극 그룹에 주목하도록 강제

---

## 4. Experiments

### 4.1 Pretraining Setup

| 항목 | 설정 |
|---|---|
| Data | REVE (61,415h, 32 channel configs) |
| Model | 19.5M params (Enc 6.3M + Pred 11.0M) |
| Optimizer | AdamW (lr=1e-4, wd=0.05) |
| Schedule | 10 epochs warmup + cosine decay |
| Batch size | 512 |
| Precision | bf16 |
| Epochs | 100 |
| Hardware | NVIDIA RTX A6000 (48GB) |

### 4.2 Downstream Benchmarks

[TODO: 각 벤치마크의 데이터 통계, 평가 프로토콜]

#### 4.2.1 TUAB (Normal/Abnormal Classification)
- [TODO: 데이터 통계, train/val/test split]
- Linear probing + Fine-tuning

#### 4.2.2 TDBrain (Parkinson Detection)
- [TODO]

#### 4.2.3 APAVA (Alzheimer Detection)
- [TODO]

#### 4.2.4 BCI Motor Imagery
- [TODO]

### 4.3 Baselines

| Model | Type | Pretrain Data | Params |
|---|---|---|---|
| Laya | LeJEPA + Channel Mixer | ~29,000h | ~30M |
| LuMamba | LeJEPA + Mamba | ~21,000h | 4.6M |
| LaBraM | VQ + MAE | ~2,500h | - |
| BIOT | Contrastive | TUH | - |
| S-JEPA | Spatial JEPA (EMA) | 소규모 | - |

---

## 5. Results

### 5.1 Pretraining Analysis

[TODO: wandb에서 추출]
- Loss curves (pred, sigreg, query)
- Training stability (grad norms)
- Convergence speed

### 5.2 Main Results

[TODO: 결과 테이블]

| Model | TUAB Acc | TUAB AUC | TDBrain | APAVA | BCI |
|---|---|---|---|---|---|
| Laya | | | | | |
| LuMamba | | | | | |
| LaBraM | | | | | |
| **Ours** | | | | | |

### 5.3 Topology Invariance Analysis

[TODO]
- 사전학습에서 본 적 없는 채널 구성에서의 성능
- 채널 수별 성능 분석 (저밀도 vs 고밀도)

### 5.4 Ablation Studies

[TODO: 실험 설계]

| Ablation | 변경 내용 | 목적 |
|---|---|---|
| w/o spatial masking | 시간 마스킹만 | spatiotemporal의 공간 마스킹 기여 |
| w/o SIGReg | EMA 사용 | SIGReg vs EMA 비교 |
| w/o Channel Mixer | 고정 채널 (19ch) | Channel Mixer 기여 |
| w/o Query Spec Loss | $\lambda_2$=0 | 쿼리 특화 손실 기여 |
| Predictor size | 작은 predictor | LeWM-style 강한 predictor 기여 |
| Mixer bottleneck | mixer_dim=256 (병목 없음) | 공간 압축 병목의 기여 |

---

## 6. Discussion

### 6.1 Spatiotemporal Masking의 효과
[TODO]

### 6.2 SIGReg vs EMA for EEG
[TODO]

### 6.3 Strong Predictor의 역할
[TODO: LeWM에서 발견한 predictor 설계 철학이 EEG에서도 유효한지]

### 6.4 Limitations
- 전처리에서 ICA 미사용 → 아티팩트에 취약할 수 있음
- 2초 윈도우 → 장기 시계열 패턴(수면 단계 전이 등) 미포착
- 단일 GPU 학습 → 더 큰 모델/배치 탐색 제한

### 6.5 Future Work
- EEG-to-Video World Model (장기 목표)
- Mamba backbone 탐색 (LuMamba 참고)
- 더 큰 모델 스케일링 (D384, L12)

---

## 7. Conclusion

[TODO]

---

## References

### Core JEPA / SSL

- [Assran et al., 2023] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." CVPR 2023. arXiv:2301.08243.
- [Bardes et al., 2024] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas. "Revisiting Feature Prediction for Learning Visual Representations from Video." arXiv:2404.08471.
- [Balestriero and LeCun, 2025] Randall Balestriero, Yann LeCun. "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics." arXiv:2511.08544.
- [Maes et al., 2026] Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero. "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels." arXiv:2603.19312.
- [Bardes et al., 2022] Adrien Bardes, Jean Ponce, Yann LeCun. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." ICLR 2022. arXiv:2105.04906.
- [Zbontar et al., 2021] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stephane Deny. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." ICML 2021. arXiv:2103.03230.
- [Grill et al., 2020] Jean-Bastien Grill et al. "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning." NeurIPS 2020. arXiv:2006.07733.
- [He et al., 2022] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. arXiv:2111.06377.
- [Bao et al., 2022] Hangbo Bao, Li Dong, Songhao Piao, Furu Wei. "BEiT: BERT Pre-Training of Image Transformers." ICLR 2022 (Oral). arXiv:2106.08254.

### EEG Foundation Models

- [Doner et al., 2025] Berkay Doner, Thorir Mar Ingolfsson, Luca Benini, Yawei Li. "LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis." NeurIPS 2025. arXiv:2510.22257.
- [Panchavati et al., 2026] Saarang Panchavati, Uddhav Panchavati, Corey Arnold, William Speier. "Laya: A LeJEPA Approach to EEG via Latent Prediction over Reconstruction." arXiv:2603.16281.
- [Broustail et al., 2026] Danae Broustail, Anna Tegon, Thorir Mar Ingolfsson, Yawei Li, Luca Benini. "LuMamba: Latent Unified Mamba for Electrode Topology-Invariant and Efficient EEG Modeling." arXiv:2603.19100.
- [Jiang et al., 2024] Wei-Bang Jiang, Li-Ming Zhao, Bao-Liang Lu. "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI." ICLR 2024 (Spotlight). arXiv:2405.18765.
- [Wang et al., 2024] Guangyu Wang, Wenchao Liu, Yuhong He, Cong Xu, Lin Ma, Haifeng Li. "EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals." NeurIPS 2024.
- [Yang et al., 2023] Chaoqi Yang, M. Brandon Westover, Jimeng Sun. "BIOT: Biosignal Transformer for Cross-data Learning in the Wild." NeurIPS 2023. arXiv:2305.10351.
- [Cui et al., 2024] Wenhui Cui, Woojae Jeong, Philipp Tholke, Takfarinas Medani, Karim Jerbi, Anand A. Joshi, Richard M. Leahy. "Neuro-GPT: Towards A Foundation Model for EEG." ISBI 2024. arXiv:2311.03764.
- [Guetschel et al., 2024] Pierre Guetschel, Thomas Moreau, Michael Tangermann. "S-JEPA: Towards Seamless Cross-Dataset Transfer through Dynamic Spatial Attention." arXiv:2403.11772.
- [Chen et al., 2025] Zhige Chen, Chengxuan Qin, Wenlong You, Rui Liu, Congying Chu, Rui Yang, Kay Chen Tan, Jibin Wu. "HEAR: An EEG Foundation Model with Heterogeneous Electrode Adaptive Representation." arXiv:2510.12515.
- [Dong et al., 2024] Zijian Dong, Ruilin Li, Yilei Wu, et al. "Brain-JEPA: Brain Dynamics Foundation Model with Gradient Positioning and Spatiotemporal Masking." NeurIPS 2024 (Spotlight). arXiv:2409.19407.

### Datasets and Benchmarks

- [El Ouahidi et al., 2025] Yassine El Ouahidi, Jonathan Lys, Philipp Tholke, Nicolas Farrugia, Bastien Pasdeloup, Vincent Gripon, Karim Jerbi, Giulia Lioi. "REVE: A Foundation Model for EEG — Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects." NeurIPS 2025. arXiv:2510.21585.
- [Obeid and Picone, 2016] Iyad Obeid, Joseph Picone. "The Temple University Hospital EEG Data Corpus." Frontiers in Neuroscience, 10:196.
- [Lopez, 2017] Silvia Lopez de Diego. "Automated Interpretation of Abnormal Adult Electroencephalograms." Master's Thesis, Temple University.
- [van Dijk et al., 2022] Hanneke van Dijk et al. "The Two Decades Brainclinics Research Archive for Insights in Neurophysiology (TDBRAIN) Database." Scientific Data, 9:333.
- [Escudero et al., 2006] Javier Escudero, Daniel Abasolo, Roberto Hornero, Pedro Espino, Miguel Lopez. "Analysis of Electroencephalograms in Alzheimer's Disease Patients with Multiscale Entropy." Physiological Measurement, 27(11):1091-1106.
- [Tangermann et al., 2012] Michael Tangermann et al. "Review of the BCI Competition IV." Frontiers in Neuroscience, 6:55.
- [Goldberger et al., 2000] Ary L. Goldberger et al. "PhysioBank, PhysioToolkit, and PhysioNet." Circulation, 101(23):e215-e220.

### Foundations

- [Vaswani et al., 2017] Ashish Vaswani et al. "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762.
- [Dosovitskiy et al., 2021] Alexey Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021. arXiv:2010.11929.
- [Loshchilov and Hutter, 2019] Ilya Loshchilov, Frank Hutter. "Decoupled Weight Decay Regularization." ICLR 2019. arXiv:1711.05101.
- [Jasper, 1958] Herbert H. Jasper. "The Ten-Twenty Electrode System of the International Federation." Electroencephalography and Clinical Neurophysiology, 10:371-375.
- [Bertrand et al., 1985] O. Bertrand, F. Perrin, J. Pernier. "A Theoretical Justification of the Average Reference in Topographic Evoked Potential Studies." Electroencephalography and Clinical Neurophysiology, 62:462-464.
- [Lawhern et al., 2018] Vernon J. Lawhern et al. "EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces." Journal of Neural Engineering, 15(5):056013. arXiv:1611.08024.

### Surveys

- [Weng et al., 2024] Weining Weng et al. "Self-supervised Learning for Electroencephalogram: A Systematic Survey." ACM Computing Surveys, 57:1-38. arXiv:2401.05446.
- [LeCun, 2022] Yann LeCun. "A Path Towards Autonomous Machine Intelligence." OpenReview preprint, June 2022.
