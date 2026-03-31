# Research Method

> 최종 업데이트: 2026-03-31
> 버전: v1.1

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
| 2 | 샘플링 레이트 (100~2000Hz) | 200Hz로 리샘플링 | scipy.signal.resample |
| 3 | 참조 전극 (Reference) | Common Average Reference로 통일 | 전 채널 평균 차감 |
| 4 | 노이즈/아티팩트 | Bandpass 필터 0.5~75Hz + JEPA 자체 강건성 | 최소 전처리 트렌드 |
| 5 | 세그멘테이션 | 2초 고정 윈도우 | Laya/LuMamba와 동일 조건 |

### 전처리 파이프라인

오프라인 전처리와 온라인 처리를 분리하여 유연성과 효율을 확보.

```
[오프라인 — 1회성]
원본 EEG (.edf)
    ↓
① 채널명 표준화 (Alias 매핑, REVE Position Bank 543개 전극)
② 유효 좌표 없는 채널 드롭 (드롭 비율 로깅, 50% 미만 데이터셋 제외)
③ 리샘플링 → 200Hz
④ Bandpass 필터 → 0.5~75Hz
⑤ 2초 윈도우 분할 → LitData 저장 (샘플 단위 메타데이터 포함)

[온라인 — 학습 시]
⑥ CAR 적용 (Dataset.__getitem__ 내, 패딩 전이므로 마스킹 불필요)
⑦ Collate: 배치 내 max_c로 zero-padding + padding_mask 생성
    ↓
[B, max_c, 400] + padding_mask [B, max_c] → Channel Mixer
```

상세 설계: [data_pipeline.md](data_pipeline.md) 참조

---

## Stage 2: 자기지도 사전학습 (Self-Supervised Pretraining)

라벨 없이 대량의 EEG 데이터로 범용 인코더를 학습하는 핵심 단계.

### 아키텍처 구성요소

```
[B × C × 400]  ← Stage 1 출력
    ↓
┌─────────────────────────────────────┐
│ Patch Embedder                      │
│ - Depthwise 1D Conv (채널별 독립)     │
│ - kernel=patch_size, stride=patch_size│
│ → [B × C × N × E]                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Dynamic Channel Mixer               │
│ - 전극 3D 좌표 → Fourier Feature 인코딩│
│ - Nq개 Learned Query로 Cross-Attention│
│ - Query Specialization Loss 적용     │
│ → [B × N × D]  (고정 크기)           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Spatiotemporal Masking               │
│ - 60% 마스킹 비율                     │
│ - 시간축: 500ms~1s 연속 블록           │
│ - 공간축: 채널 그룹 단위 마스킹         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Transformer Encoder                  │
│ - RoPE positional embedding          │
│ - 2-pass: full sequence → context-only│
│ → Z (full), Z_ctx (context)         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Predictor                            │
│ - 마스킹된 영역의 잠재 벡터 예측        │
│ → ẑ_target                          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Loss Function                        │
│ - L_pred: MSE(ẑ_target, z_target)   │
│ - L_sigreg: SIGReg(Z)               │
│ - L_query: Query Specialization      │
│ - L_total = L_pred + λ₁·L_sigreg    │
│            + λ₂·L_query             │
└─────────────────────────────────────┘
```

### 핵심 설계 결정

| 결정 사항 | 채택 | 대안 (기각) | 이유 |
|---|---|---|---|
| 붕괴 방지 | SIGReg (LeWM) | EMA + Stop-gradient | 수학적으로 명확, 하이퍼파라미터 적음 |
| 잠재 공간 | 연속적 (Continuous) | 이산적 RVQ (DELTA) | SIGReg와 수학적 비호환 |
| 마스킹 | 시공간 동시 | 시간만 (Laya) / 공간만 (S-JEPA) | 차별화 포인트 |
| 채널 처리 | Learned-query Mixer | 고정 채널 선택 / ID 임베딩 | 임의 몽타주 대응 |
| 윈도우 크기 | 2초 | 5초, 10초 | Baseline과 공정 비교 |

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
| ② | **TDBrain** | 파킨슨 탐지 | 26ch | Accuracy, AUROC | LuMamba | ❌ TUH 무관 |
| ③ | **APAVA** | 알츠하이머 탐지 | 다름 | Accuracy, AUROC | LuMamba | ❌ TUH 무관 |
| ④ | **BCI (MI)** | 운동 상상 분류 | 22~64ch | Accuracy | S-JEPA | ❌ TUH 무관 |

- ①: 표준 벤치마크로 전체 baseline과 직접 비교
- ②③: TUH와 무관 + 다른 몽타주 → **Topology-Invariant 전이 능력 증명**
- ④: BCI 실용성 + TUH와 완전 무관 → **leakage 논쟁 완전 차단**

---

## 논문 기여도 (Expected Contributions)

1. **Channel Mixing + Spatiotemporal Masking의 최초 결합**: 공간적 이질성과 시공간 역학을 동시에 해결
2. **SIGReg 기반 EEG 사전학습**: EMA 없이 안정적인 JEPA 학습을 EEG 도메인에서 검증
3. **최소 전처리로 SOTA 달성**: 복잡한 신호 정제 없이 강건한 분류 성능 증명
4. **진정한 Topology-Invariant 학습**: 채널 수를 인위적으로 조작하지 않고, padding_mask 기반으로 가변 채널을 있는 그대로 처리 (REVE+TUH 6TB 혼합 사전학습)
