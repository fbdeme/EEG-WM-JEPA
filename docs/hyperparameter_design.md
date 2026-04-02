# Hyperparameter Design

> 최종 업데이트: 2026-04-01
> 상태: 모델 구조 확정, 학습 파라미터 조정 중

---

## 1. 배경 조사: 레퍼런스 논문 하이퍼파라미터 비교

우리 모델(EEG-WM-JEPA)은 3가지 레퍼런스를 결합한 구조:
- **LeWM**: base model (SIGReg + Predictor 아키텍처)
- **Laya/LUNA**: Channel Mixer + Patch Embedder
- **Brain-JEPA**: Spatiotemporal Masking 전략

### 레퍼런스 논문별 주요 하이퍼파라미터

| 파라미터 | LeWM | Laya | LUNA-Base | LaBraM | Brain-JEPA(B) |
|---|---|---|---|---|---|
| Encoder dim | 192 | 384 | 256 | 200 | 768 |
| Encoder depth | 12 | 12 | 8 | 12 | 12 |
| Encoder heads | 3 | 6 | 8 | 10 | 12 |
| Predictor depth | 6 | 4 | - | - | 6 |
| Predictor inner dim | 1024 (16h×64dh) | 128 proj | - | - | 384 |
| Patch size | 14px (이미지) | 25 (125ms) | 40 (200ms) | 200 (1s) | 16 |
| Num queries | - | 16 | 4 | - | - |
| Mixer dim | - | 32 | 64 | - | - |
| Mask ratio | - | 0.6 | 0.5 | 0.8 | varied |
| LR | 5e-5 | 1e-4 | 1.25e-4 | 5e-4 | 1e-3 peak |
| Weight decay | 1e-3 | 0.05 | 0.05 | 0.05 | 0.04→0.4 |
| Batch size | 128 | 256 | 8160 | 64 | 512 |
| SIGReg lambda | 0.09 | 0.05 | - | - | - |
| SIGReg projections | 1024 | - | - | - | - |
| Query spec weight | - | 1.0 | 0.8 | - | - |
| Total params | ~15M | ~30M | 7M | - | 86M |
| Pred/Enc 비율 | **~2.0** | ~0.33 | - | - | ~0.08 |

### 핵심 발견: LeWM의 독특한 설계 철학

LeWM은 다른 JEPA 모델과 달리 **predictor가 encoder보다 2배 큰** 구조:
- Encoder (ViT-Tiny): 5M params (D192, L12, H3)
- Predictor: 10M params (D192, L6, H16, dim_head=64)

이유: LeWM은 EMA(target encoder) 없이 SIGReg로 collapse를 방지. 이 방식에서는 predictor가 충분히 강력해야 encoder에게 좋은 representation을 학습하도록 압력을 줄 수 있음. 약한 predictor는 평균값만 예측하는 shortcut에 빠질 위험.

**predictor inner dim의 비밀**: LeWM predictor의 working dim은 encoder와 동일한 192. 하지만 attention 내부에서 `heads(16) × dim_head(64) = 1024`로 확장 → `to_qkv: Linear(192→3072)`, `to_out: Linear(1024→192)`. 이 내부 확장이 predictor 파라미터의 대부분을 차지.

---

## 2. 조정 대상 파라미터 전체 목록

### A. 모델 구조 (학습 시작 전 확정 필요)

| 카테고리 | 파라미터 | 설명 |
|---|---|---|
| **Patch Embedder** | patch_size | 시간 해상도 결정 (패치 수 = 400 / patch_size) |
| | patch_embed_dim | 채널별 패치 임베딩 차원 |
| **Channel Mixer** | num_queries | 학습 쿼리 수 (공간 해상도) |
| | mixer_dim | 믹서 출력 차원 (공간 압축 강도) |
| | fourier_features | 3D 좌표 푸리에 인코딩 차원 |
| | num_heads_mixer | 크로스어텐션 헤드 수 |
| **Encoder** | encoder_dim | 메인 representation 차원 |
| | encoder_depth | Transformer 레이어 수 |
| | encoder_heads | 어텐션 헤드 수 |
| | dropout | 드롭아웃 비율 |
| **Predictor** | predictor_depth | Predictor 레이어 수 |
| | predictor_heads | 어텐션 헤드 수 |
| | predictor_dim_head | 고정 dim_head (LeWM 스타일) |
| | predictor_mlp_dim | MLP 히든 차원 |
| **Projector** | projector_hidden | Projector MLP 히든 차원 |
| | projector_out | Projector 출력 차원 |

### B. 학습 파라미터

| 카테고리 | 파라미터 | 설명 |
|---|---|---|
| **Masking** | mask_ratio | 마스킹 비율 |
| | mask_block_min/max_sec | 마스크 블록 크기 |
| **Loss** | lambda_sigreg | SIGReg 정규화 강도 |
| | lambda_query | 쿼리 특화 손실 가중치 |
| **SIGReg** | sigreg_num_projections | 랜덤 프로젝션 수 |
| **Optimizer** | lr | 학습률 |
| | weight_decay | L2 정규화 |
| | batch_size | 배치 크기 |
| | epochs | 학습 에포크 수 |

---

## 3. 파라미터별 주요 내용

### Encoder Depth / Dim / Heads

- **depth**: Transformer 블록 수. 깊을수록 장거리 의존성 캡처. EEG에서 수 초에 걸친 oscillation 패턴(alpha, beta 리듬) 학습에 충분한 깊이 필요. 레퍼런스 최소 8(LUNA) ~ 12(나머지 전부)
- **dim**: 토큰 표현 벡터 크기. 192(LeWM) ~ 768(Brain-JEPA). 클수록 표현력 풍부하지만 파라미터 급증
- **heads**: Multi-head attention 헤드 수. 각 헤드가 다른 관점(주파수 패턴, 시간적 인접성 등)에서 토큰 관계를 봄. 보통 dim_head=64가 표준

### Predictor Depth / Heads / dim_head

- JEPA에서 마스킹된 영역의 latent representation을 예측하는 네트워크
- 너무 얕으면 trivial solution, 너무 강하면 encoder 학습 동기 감소
- **LeWM 특이점**: dim_head를 64로 고정하고 heads를 늘려서 attention 내부 inner_dim을 키움. `heads × dim_head`가 encoder_dim보다 훨씬 큼 → 이것이 predictor 파라미터의 대부분
- SIGReg 방식에서는 predictor 용량이 representation 품질에 직결 (EMA 방식과 반대)

### Patch Size

- 원시 EEG를 몇 샘플씩 묶어 토큰화. 200Hz 기준: 40→200ms, 25→125ms
- 패치 수 = Transformer sequence length의 한 축 (seq = queries × patches)
- 적을수록(큰 패치): 연산 효율적이지만 마스킹이 거침
- 많을수록(작은 패치): 세밀한 마스킹 가능하지만 attention O(n²) 증가
- EEG alpha 리듬(8-13Hz, 주기 80-125ms) 고려 시 125ms(patch=25)가 1주기를 커버

### Channel Mixer (num_queries, mixer_dim)

- **num_queries**: 고정 수의 learned query가 가변 채널 정보를 cross-attention으로 수집. 각 쿼리가 특정 뇌 영역을 담당하도록 query specialization loss로 유도
- **mixer_dim**: 채널 믹서 출력 차원. **정보 병목(bottleneck)**으로 동작해야 함 — encoder_dim과 같으면 병목이 없어 공간 압축 학습이 약해짐. Laya=32, LUNA=64로 매우 좁음
- mixer_dim ≠ encoder_dim일 때 Transformer의 input_proj가 자동으로 Linear(mixer_dim → encoder_dim) 수행

### SIGReg Lambda / Projections

- **lambda**: JEPA의 representation collapse 방지 강도. 분포가 등방성 가우시안에 가깝도록 강제. 너무 크면 정규화가 prediction loss를 압도, 너무 작으면 collapse 위험. LeWM: [0.01, 0.2] 범위 안정, 최적 0.09
- **num_projections**: 고차원 분포의 가우시안성을 랜덤 1D 프로젝션으로 추정. 많을수록 정확하지만 연산 증가. LeWM=1024, 최소 300+ 권장

### Query Specialization Weight

- 각 쿼리의 attention weight 분포를 비교해서 서로 다른 채널에 주목하도록 강제
- 없으면 모든 쿼리가 같은 채널만 보는 degenerate solution 위험
- LUNA=0.8, Laya=1.0. 충분한 다양성 압력 필요

### Batch Size

- SIGReg에 특히 중요: 배치 내 샘플 분포로 가우시안 통계를 추정하므로 큰 배치가 유리
- gradient accumulation으로 effective batch size 확보 가능

---

## 4. 확정된 파라미터 (모델 구조)

### 변경 내역

| 파라미터 | 변경 전 | 변경 후 | 근거 |
|---|---|---|---|
| **patch_size** | 40 (200ms, 10패치) | **25** (125ms, 16패치) | Laya 기준, alpha 리듬 1주기 커버, 세밀한 마스킹 |
| **mixer_dim** | 256 (병목 없음) | **64** (4x 압축) | 공간 압축 병목 형성, LUNA-Base(64) 참고 |
| **encoder_depth** | 6 | **8** | LUNA-Base 수준, 레퍼런스 최소 8 |
| **predictor_depth** | 3 | **6** | LeWM과 동일, 충분한 predictor 용량 |
| **predictor_heads** | 4 (dh=64, inner=256) | **12** (dh=64, inner=768) | LeWM 스타일: 고정 dim_head + 많은 heads로 inner dim 확장 |
| **predictor_dim_head** | encoder_dim/heads (가변) | **64** (고정) | LeWM과 동일, attention 내부 확장의 핵심 |
| **predictor_mlp_dim** | encoder_dim×4=1024 | **2048** | LeWM과 동일 |
| **projector_hidden** | 1024 | **2048** | LeWM과 동일 |
| **projector_out** | 128 | **256** | encoder_dim과 일치 (LeWM 패턴) |

### 확정된 전체 모델 구조

```
Patch Embedder: patch_size=25, embed_dim=64
Channel Mixer:  queries=8, mixer_dim=64, fourier=32, heads=4
Encoder:        D=256, L=8, H=4, dim_head=64, mlp=1024, dropout=0.1
Predictor:      D=256, L=6, H=12, dim_head=64, inner=768, mlp=2048
Projector:      256 → 2048 → 256 (BatchNorm)

Sequence length: 8 queries × 16 patches = 128 tokens
Total params:    19.5M (Encoder 6.3M + Predictor 11.0M + 기타 2.2M)
Pred/Enc 비율:   1.75 (LeWM ~2.0 근사)
```

### GPU 메모리 프로파일 (RTX A6000, 48GB)

| Batch Size | 19ch | 64ch | 128ch |
|---|---|---|---|
| 256 | 1.1GB | 1.6GB | 2.0GB |
| 512 | 2.2GB | 2.9GB | 3.8GB |
| 1024 | 4.0GB | 5.4GB | 7.3GB |

→ 메모리 제약 없음. batch_size 512 이상 가능.

---

## 5. 남은 파라미터 (학습 시작 전 확정 필요)

### 조정 필요 (레퍼런스 대비 불일치 발견)

| 파라미터 | 현재 값 | 레퍼런스 범위 | 상태 |
|---|---|---|---|
| **lambda_sigreg** | ~~1.0~~ → 0.09 | 0.05~0.09 | config 반영 완료, 검증 필요 |
| **lambda_query** | ~~0.1~~ → 0.8 | 0.8~1.0 | config 반영 완료, 검증 필요 |
| **sigreg_num_projections** | ~~64~~ → 512 | 300~1024 | config 반영 완료, 검증 필요 |
| **batch_size** | ~~64~~ → 512 | 256~8160 | config 반영 완료, 실제 학습 시 VRAM 확인 필요 |

### 현재 값 유지 (레퍼런스와 일치)

| 파라미터 | 현재 값 | 레퍼런스 | 비고 |
|---|---|---|---|
| mask_ratio | 0.6 | 0.5~0.6 | Laya/LuMamba 표준 |
| mask_block_min/max | 0.5~1.0초 | 비슷 | 패치 수 변경으로 블록 크기 재확인 필요 |
| lr | 1e-4 | 5e-5~1.25e-4 | 범위 내 |
| weight_decay | 0.05 | 0.05 | 표준 |
| epochs | 100 | 50~300 | 적절 |
| dropout | 0.1 | 0.1 | 표준 |

### 추가 확정 (2026-04-01 후반)

| 파라미터 | 변경 전 | 변경 후 | 근거 |
|---|---|---|---|
| **precision** | fp16 + GradScaler | **bf16** (GradScaler 제거) | 레퍼런스 전부 bf16, 범위가 fp32와 동일하여 SIGReg overflow 방지 |
| **warmup** | min(total_steps//10, 1000) = ~2 epochs | **10 epochs** (epoch 기반) | LUNA=10/60(17%), LaBraM=5/50(10%), ~10% 표준 |
| **mask_block** | 0.5~1.0초 (4~8패치) | **유지** | 16패치 중 4~8패치 블록. spatiotemporal 마스킹에서 시간 블록이 너무 크면 공간 마스킹 비중 감소. Laya(5~10)와 겹치는 범위 |
| **예측 방식** | masking (A) | **masking 유지** | LeWM의 autoregressive는 프레임 간 예측. 우리의 spatiotemporal masking이 논문 기여의 핵심이므로 masking 방식 유지 |

### 학습 시 검증 필요

- **grad_accum**: batch=512가 실제 스트리밍에서 OOM이면 올려야 함
- **SIGReg projections 512 vs 1024**: 512로 시작 후 안정성 보고 판단

---

## 로깅 체계

wandb 기반 실시간 모니터링 구축 완료 (2026-04-02 확장):

**매 Step 기록:**
- `train/loss`, `train/pred_loss`, `train/sigreg_loss`, `train/query_loss`
- `train/grad_norm`, `train/lr`
- `train/batch_max_channels`, `train/batch_mean_channels`
- `diag/emb_mean_var`, `diag/emb_min_var` — SIGReg 내부: embedding 분산 (min_var→0이면 collapse 징후)
- `diag/mean_abs_corr` — SIGReg 내부: 차원 간 평균 상관 (→0이면 등방성 달성)
- `diag/actual_mask_ratio` — 실제 마스킹 비율 (설정 60%와 비교)
- `perf/samples_per_sec` — 학습 throughput

**매 10 Step 기록:**
- `grad/encoder`, `grad/predictor`, `grad/channel_mixer`, `grad/projector` — 컴포넌트별 grad norm (Pred/Enc 비율 설계 검증)

**매 Epoch 기록:**
- `epoch/loss`, `epoch/pred_loss`, `epoch/sigreg_loss`, `epoch/query_loss`
- `epoch/duration_sec`
- `perf/gpu_mem_allocated_gb`, `perf/gpu_mem_reserved_gb`

**체크포인트:**
- 5 epoch마다 저장
- wandb Artifact 업로드
- HF public model repo 업로드 (`checkpoints/{run_name}/epoch_XXXX.pt` 구조)
- 로컬은 최신 3개만 유지 (디스크 절약)
- 실험 마무리 후 HF repo에 best 모델만 남기고 정리

**분석 관점:**
- pred_loss 하강 추이 → 모델이 latent prediction을 학습하고 있는지
- sigreg_loss 안정성 → collapse 없이 representation이 분산되는지
- emb_min_var → collapse 조기 경보 (loss보다 민감)
- mean_abs_corr → 등방성 가우시안 달성 여부
- query_loss → 쿼리들이 서로 다른 뇌 영역을 담당하는지
- grad/encoder vs grad/predictor → Predictor가 encoder를 압도하지 않는지
- grad_norm → 학습 안정성, 폭발/소실 감지
- batch channel 통계 → 데이터 다양성 모니터링
- actual_mask_ratio → spatiotemporal masking이 설정대로 작동하는지

---

## 학습 인프라 전략 (2026-04-02 확정)

### 병목 분석

| 구간 | 시간/step | 비중 |
|------|----------|------|
| GPU 연산 (A6000) | 0.27s | - |
| HF 스트리밍 로딩 | 1.8s (avg), 48s (max) | 85% |

→ HF 스트리밍은 GPU 가동률 15%. 로컬 디스크 필요하나 full 데이터 ~2.5TB > 서버 422GB.

### 더블버퍼 데이터 로더 (채택)

```
[다운로드 스레드 8개] → [버퍼 A: 30GB] → [GPU 학습]
                        [버퍼 B: 30GB] ← (학습 끝나면 삭제)
```

- 버퍼 A 학습 중 → 8 스레드가 버퍼 B를 병렬 다운로드
- 버퍼 A 학습 완료 → 삭제, 버퍼 B로 전환
- 8 병렬 DL 속도 (1.8s/shard) < GPU 학습 속도 (2.6s/shard) → **GPU가 병목 = GPU 100% 활용**
- 디스크 사용: ~60GB (버퍼 2개)
- 로컬 2.5TB 저장과 학습 시간 동일 (GPU가 병목이므로)

### 학습 계획

| 단계 | 데이터 | Epoch | 소요 시간 |
|------|--------|-------|----------|
| HP 튜닝 | 로컬 300GB (부분) | 10-20 | 실험당 2-4시간 |
| Config 검증 | 로컬 300GB | 50 | 12시간 |
| Final 학습 | 더블버퍼 full | 50-100 | 5-10일 |

### 실측 성능 (A6000 48GB, batch=512)

| 항목 | 값 |
|------|-----|
| Step time (로컬) | 0.27s |
| VRAM 사용 | 5.17GB |
| HF download | 36MB/s (단일), 8 병렬 시 shard당 1.8s |
| Full 데이터 | ~3,319 shards, ~16.6M windows, ~2.5TB |
| Epoch time (full) | ~2.4시간 |
