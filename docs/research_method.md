# Research Method

> 최종 업데이트: 2026-04-01
> 버전: v2.0 (하이퍼파라미터 확정, 학습 인프라 구축 완료)

## 연구 제목 (가제)

**"Topology-Invariant EEG Foundation Model with Spatiotemporal JEPA and SIGReg Regularization"**

---

## 전체 파이프라인 개요

```
Stage 1: EEG 표준화 → Stage 2: 자기지도 사전학습 → Stage 3: 다운스트림 평가
```

---

## Stage 1: EEG 표준화 (Preprocessing & Standardization)

모든 이질적인 EEG 데이터를 동일한 형태로 변환하는 단계.

### 표준화 대상 5가지 요소

| # | 이질성 요소 | 해결 방법 | 비고 |
|---|---|---|---|
| 1 | 채널 수/위치 (8~256ch) | 3D 좌표 Fourier 인코딩 + Learned-query Cross-Attention (Stage 2 Channel Mixer에서 처리) | LUNA/Laya 방식 |
| 2 | 샘플링 레이트 (100~2000Hz) | 200Hz로 리샘플링 | REVE는 이미 200Hz |
| 3 | 참조 전극 (Reference) | Common Average Reference로 통일 | 온라인 적용 (패딩 전) |
| 4 | 노이즈/아티팩트 | Bandpass 필터 0.5~75Hz + JEPA 자체 강건성 | 최소 전처리 트렌드 |
| 5 | 세그멘테이션 | 2초 고정 윈도우 | Laya/LuMamba와 동일 조건 |

### 전처리 파이프라인

오프라인 전처리와 온라인 처리를 분리하여 유연성과 효율을 확보.

```
[오프라인 — 1회성, scripts/preprocess_reve.py]
REVE 원본 (HuggingFace, brain-bzh/reve-dataset)
    ↓
① 세션 분리 + 플래그된 세션/채널 제거 (df_corrected 기반)
② CAR 적용 (Common Average Reference)
③ Bandpass 필터 → 0.5~75Hz
④ Clip → ±15 (REVE 기준)
⑤ 2초 윈도우 분할 (400 samples)
    ↓
HuggingFace Private repo (fbdeme/reve-preprocessed) 업로드
    ↓
레코딩별 HF 캐시 자동 삭제 (디스크 절약)

[온라인 — 학습 시, StreamingEEGDataset]
HF에서 parquet 스트리밍 로드
    ↓
Collate: 배치 내 max_c로 zero-padding + padding_mask 생성
    ↓
[B, max_c, 400] + padding_mask [B, max_c] → Channel Mixer
```

### 사전학습 데이터: REVE 단일 소스

| 항목 | 내용 |
|---|---|
| 데이터셋 | REVE (brain-bzh/reve-dataset) |
| 총 규모 | 61,415시간, ~24,260명, 92개 데이터셋 |
| TUH 포함 | 26,847시간 (44%) — 별도 확보 불필요 |
| 채널 분포 | 82% 저밀도(≤24ch), 15% 고밀도(125~129ch) |
| 채널 구성 | 32종 — Channel Mixer의 topology-invariant 학습에 최적 |

상세 설계: [data_pipeline.md](data_pipeline.md) 참조

---

## Stage 2: 자기지도 사전학습 (Self-Supervised Pretraining)

라벨 없이 대량의 EEG 데이터로 범용 인코더를 학습하는 핵심 단계.

### 아키텍처 구성요소

```
[B × C × 400]  ← Stage 1 출력 (C는 가변)
    ↓
┌─────────────────────────────────────┐
│ Patch Embedder                      │
│ - Depthwise 1D Conv (채널별 독립)     │
│ - patch_size=25 (125ms), 16 patches │
│ → [B × C × 16 × 64]               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Dynamic Channel Mixer               │
│ - 전극 3D 좌표 → Fourier Feature(32) │
│ - 8개 Learned Query, mixer_dim=64   │
│ - 4x 공간 압축 병목 (64 → 256 proj) │
│ - padding_mask로 가짜 채널 차단       │
│ → [B × 128 × 256]  (8q × 16p)     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Spatiotemporal Masking               │
│ - 60% 마스킹 비율                     │
│ - 시간축: 0.5~1.0초 연속 블록 (4~8패치)│
│ - 공간축: 쿼리 그룹 단위 마스킹         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Transformer Encoder (6.3M)          │
│ - D=256, L=8, H=4 (dim_head=64)    │
│ - MLP=1024, dropout=0.1            │
│ → Z_ctx (context representations)  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Predictor (11.0M, LeWM style)       │
│ - D=256, L=6, H=12, dim_head=64    │
│ - inner_dim=768, MLP=2048           │
│ - Pred/Enc ratio = 1.75 (LeWM ≈ 2.0)│
│ → ẑ_target (predicted latents)     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Projector (MLP + BatchNorm)         │
│ - 256 → 2048 → 256                 │
│ - Encoder/Predictor 출력 각각 투영    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Loss Function                        │
│ - L_pred: MSE(ẑ_target, z_target)   │
│ - L_sigreg: SIGReg(Z), λ=0.09      │
│ - L_query: Query Spec, λ=0.8        │
│ - L_total = L_pred + 0.09·L_sigreg  │
│            + 0.8·L_query            │
└─────────────────────────────────────┘
```

### 모델 사양 요약

| 컴포넌트 | 파라미터 | 비율 |
|---|---|---|
| Encoder | 6.3M | 32.4% |
| Predictor | 11.0M | 56.5% |
| Projector ×2 | 2.1M | 10.8% |
| Channel Mixer | 0.06M | 0.3% |
| Patch Embedder | 0.002M | 0.0% |
| **Total** | **19.5M** | 100% |

### 핵심 설계 결정

| 결정 사항 | 채택 | 대안 (기각) | 이유 |
|---|---|---|---|
| 붕괴 방지 | SIGReg (LeWM) | EMA + Stop-gradient | 수학적으로 명확, 하이퍼파라미터 적음 |
| 잠재 공간 | 연속적 (Continuous) | 이산적 RVQ (DELTA) | SIGReg와 수학적 비호환 |
| 예측 방식 | **Masking** (가려진 패치 예측) | Autoregressive (미래 예측, LeWM 원본) | spatiotemporal masking이 논문 핵심 기여. AR은 파이프라인(독립 2초 윈도우) 비호환 |
| 마스킹 | 시공간 동시 | 시간만 (Laya) / 공간만 (S-JEPA) | 차별화 포인트 |
| 채널 처리 | Learned-query Mixer (병목 64→256) | 고정 채널 선택 / ID 임베딩 | 임의 몽타주 대응 + 공간 압축 |
| Predictor 설계 | 강한 Predictor (Pred/Enc=1.75) | 작은 Predictor (I-JEPA) | SIGReg 방식에서 predictor 용량이 representation 품질에 직결 (LeWM 분석) |
| 윈도우 크기 | 2초 | 5초, 10초 | Baseline과 공정 비교 |

### 학습 설정

| 항목 | 설정 | 근거 |
|---|---|---|
| Optimizer | AdamW | 표준 |
| Learning rate | 1e-4 | Laya=1e-4, LUNA=1.25e-4 |
| Weight decay | 0.05 | Laya/LUNA/LaBraM 표준 |
| Batch size | 512 | SIGReg 분포 추정에 큰 배치 유리 |
| Epochs | 100 | LeWM=100 |
| Warmup | 10 epochs (cosine decay) | LUNA=10/60, LaBraM=5/50, ~10% 표준 |
| Precision | **bf16** (no GradScaler) | 레퍼런스 전부 bf16, SIGReg overflow 방지 |
| Grad clipping | max_norm=1.0 | 표준 |
| SIGReg lambda | 0.09 | LeWM 최적 [0.01~0.2] |
| Query spec lambda | 0.8 | LUNA=0.8, Laya=1.0 |
| SIGReg projections | 512 | LeWM=1024, 학습 후 조정 가능 |
| Mask ratio | 60% | Laya/LuMamba 표준 |
| Mask block | 0.5~1.0초 (4~8패치) | spatiotemporal 마스킹 균형 |

### 레퍼런스 논문 대비 하이퍼파라미터 근거

상세 비교 분석: [hyperparameter_design.md](hyperparameter_design.md) 참조

주요 참고:
- **LeWM**: SIGReg lambda, predictor 설계 (강한 predictor, dim_head=64 고정), projector 구조
- **Laya**: patch_size=25, mask_ratio=0.6, query_spec_weight=1.0, lr=1e-4
- **LUNA-Base**: encoder D256/L8, queries=4, mixer_dim=64, warmup 10 epochs

---

## Stage 3: 다운스트림 평가 (Downstream Evaluation)

### 평가 프로토콜

| 프로토콜 | 방법 | 목적 |
|---|---|---|
| **Linear Probing** | 인코더 동결 + Linear Head | 사전학습 표현 품질 평가 |
| **Fine-tuning** | 인코더 + Task Head 전체 학습 | 태스크 최대 성능 |

### 목표 벤치마크 (4종)

| # | 벤치마크 | 태스크 | 채널 | 메트릭 | 비교 대상 | TUH 관계 |
|---|---|---|---|---|---|---|
| ① | **TUAB** | 정상/비정상 분류 | 19ch | Accuracy, AUROC | Laya, LuMamba, BIOT, LaBraM | TUH 부분집합 (표준 벤치마크) |
| ② | **TDBrain** | 파킨슨 탐지 | 26ch | Accuracy, AUROC | LuMamba | TUH 무관 |
| ③ | **APAVA** | 알츠하이머 탐지 | 다름 | Accuracy, AUROC | LuMamba | TUH 무관 |
| ④ | **BCI (MI)** | 운동 상상 분류 | 22~64ch | Accuracy | S-JEPA | TUH 무관 |

- ①: 표준 벤치마크로 전체 baseline과 직접 비교
- ②③: TUH와 무관 + 다른 몽타주 → **Topology-Invariant 전이 능력 증명**
- ④: BCI 실용성 + TUH와 완전 무관 → **leakage 논쟁 완전 차단**

### Leakage 대응 전략

REVE에 TUH가 포함(44%)되어 있으므로 TUAB 평가 시 leakage 우려 존재. 대응:
1. SSL 사전학습은 라벨 미사용 → 대부분의 논문(LuMamba, BIOT)이 이를 문제로 취급하지 않음
2. 비TUH 벤치마크(②③④)에서 우수 성능 → "leakage 없는 환경에서도 우수" 증명
3. 비TUH 벤치마크는 사전학습에서 본 적 없는 채널 구성 → Topology-Invariant 핵심 증거

---

## 학습 인프라

| 항목 | 내용 |
|---|---|
| GPU | NVIDIA RTX A6000 (48GB VRAM) |
| 사전학습 데이터 | REVE → HuggingFace Private (fbdeme/reve-preprocessed) |
| 스트리밍 | HF `datasets.load_dataset(streaming=True)` |
| 로깅 | wandb (step: loss분해/grad_norm/lr/채널통계, epoch: 요약) |
| 체크포인트 | 5 epoch마다 저장 + wandb Artifact |
| 모니터링 | wandb dashboard + 터미널 모니터 스크립트 |

### VRAM 사용량 (실측)

| Batch Size | 19ch | 64ch | 128ch |
|---|---|---|---|
| 256 | 1.1GB | 1.6GB | 2.0GB |
| 512 | 2.2GB | 2.9GB | 3.8GB |
| 1024 | 4.0GB | 5.4GB | 7.3GB |

→ batch=512에서도 48GB 대비 극히 일부 사용. 메모리 제약 없음.

---

## 논문 기여도 (Expected Contributions)

1. **Channel Mixing + Spatiotemporal Masking의 최초 결합**: 공간적 이질성과 시공간 역학을 동시에 해결
2. **SIGReg 기반 EEG 사전학습**: EMA 없이 안정적인 JEPA 학습을 EEG 도메인에서 검증. LeWM의 "강한 predictor" 설계(Pred/Enc ≈ 1.75)를 EEG에 적용
3. **최소 전처리로 SOTA 달성**: 복잡한 신호 정제 없이 강건한 분류 성능 증명
4. **진정한 Topology-Invariant 학습**: 채널 수를 인위적으로 조작하지 않고, padding_mask 기반으로 가변 채널을 있는 그대로 처리 (REVE 61,415시간, 32종 채널 구성으로 사전학습)
5. **비TUH 벤치마크 3종으로 전이 능력 검증**: leakage 논쟁 차단 + 다른 몽타주에서의 일반화 증명

---

## 타임라인

| 단계 | 상태 | 비고 |
|---|---|---|
| 연구 설계 | ✅ 완료 | 아키텍처, 데이터, 벤치마크 확정 |
| 코드 구현 (모델) | ✅ 완료 | 19.5M 모델, 전 컴포넌트 forward/backward 검증 |
| 하이퍼파라미터 설계 | ✅ 완료 | LeWM/Laya/LUNA 기반, GPU 프로파일 완료 |
| 학습 인프라 | ✅ 완료 | wandb, bf16, streaming, 체크포인트 |
| REVE 전처리 | 🔄 진행 중 | 322개 레코딩 → HF 업로드 (54/322 + 재개 중) |
| 사전학습 실행 | ⬜ 대기 | 전처리 완료 후 시작 |
| 다운스트림 평가 | ⬜ 대기 | 4종 벤치마크 |
| 논문 작성 | ⬜ 대기 | |
