# Issues & Technical Decisions

> 구현 과정에서 발견된 기술적 이슈와 해결 상태를 기록합니다.

---

## Issue #1: LeWM의 Action 조건부 예측 제거

**상태: ✅ 해결됨 (2026-03-30)**

### 문제
LeWM은 로봇 제어를 위해 설계되어, Predictor가 `action` 입력을 받아 "이 행동을 하면 다음 상태가 어떻게 될지" 예측한다. EEG 사전학습에는 action 개념이 없다.

### LeWM 원래 구조
```
상태(o₁) + 행동(a₁) → Predictor(AdaLN-zero) → 다음 상태(ô₂)
```
- `ConditionalBlock`: action 임베딩이 AdaLN-zero로 Transformer에 주입됨
- `Embedder`: raw action → action embedding 변환
- `ARPredictor`: 시간축 auto-regressive 예측 (인과적 마스킹)

### 해결 방법
| LeWM 원래 | EEG-JEPA 변경 | 이유 |
|---|---|---|
| `ConditionalBlock` (AdaLN) | `Block` (일반 Transformer) | action 조건 불필요 |
| `ARPredictor` (인과적) | `EEGPredictor` (양방향 마스킹) | 앞뒤 문맥 모두 활용 |
| `Embedder` (action 인코더) | 제거 | action 자체가 없음 |

### 향후 고려사항
World Model(EEG-to-Video)로 확장 시, action 자리에 **자극(stimulus) 정보**나 **태스크 조건**을 넣는 것을 고려할 수 있음. 이 경우 `ConditionalBlock`을 다시 도입해야 함.

---

## Issue #2: Stop-gradient 사용 여부 — LeWM vs Laya의 차이

**상태: ✅ 해결됨 (2026-03-30)**

### 문제
LeWM은 target에 stop-gradient를 안 걸고 SIGReg만으로 collapse를 방지한다. 우리도 stop-grad 없이 갈 수 있는가?

### 분석: 왜 LeWM은 stop-grad가 필요 없는가
```
LeWM: 시간축 auto-regressive 예측
  t₁의 임베딩으로 → t₂의 임베딩을 예측
  입력(t₁)과 정답(t₂)이 다른 시점이라 구조적으로 분리됨
  + SIGReg가 분포 붕괴 방지 → stop-grad 불필요
```

### 분석: 왜 우리는 stop-grad가 필요한가
```
EEG-JEPA: 같은 시퀀스 내 마스킹 기반 예측
  동일 시퀀스에서 가려진 부분의 "정답"과 "예측"을 동시에 뽑음
  stop-grad 없으면 → Encoder가 모든 위치에 동일 벡터 출력하는 게 최적
  → Collapse 위험이 LeWM보다 높음
```

### 해결 방법
Laya 논문의 설계를 따라 **prediction loss의 target에만 stop-grad 적용**:
```python
projected = self.projector(encoded)       # gradient 흐름
target = projected.detach()               # pred_loss에서는 stop-grad
sigreg_loss = self.sigreg(projected)      # SIGReg에서는 gradient 흐름
```

### 미검증 사항
- SIGReg만으로 stop-grad 없이도 마스킹 JEPA가 안정적으로 학습되는지는 **실험으로 검증 필요** (Ablation Study 후보)
- Laya가 stop-grad를 사용한 것은 검증된 사실이나, SIGReg 없이 하는지 SIGReg와 함께 하는지 세부 조합은 추가 확인 필요

---

## Issue #3: Projector의 Gradient 흐름

**상태: ✅ 해결됨 (2026-03-30)**

### 문제
초기 구현에서 SIGReg를 `encoded` (projector 이전)에 적용했더니, projector의 파라미터에 gradient가 전혀 흐르지 않았다.

### 원인
```
encoded → projector → target (detach) → pred_loss
                                         ↑ stop-grad 때문에 projector로 gradient 안 흐름

encoded → SIGReg
           ↑ encoder까지만 gradient 흐름, projector는 경로에 없음
```
결과: projector의 6개 파라미터 텐서 (Linear 2개 + BatchNorm 2개)에 `.grad = None`

### 해결 방법
SIGReg를 **projected 공간** (projector 이후)에 적용:
```
encoded → projector → projected → SIGReg  ← projector에 gradient 흐름 ✅
                         ↓
                      target (detach) → pred_loss
```

### 검증
수정 후 `test_forward_pass.py`에서 모든 파라미터의 `.grad`가 정상적으로 존재함을 확인 (Backward pass ✓)

---

## Issue #4: 마스킹 JEPA와 LeWM(Auto-regressive) 예측 방식의 근본적 차이

**상태: ⚠️ 모니터링 중**

### 문제
우리는 LeWM의 코어 모듈(SIGReg, Transformer, MLP)을 가져왔지만, **예측 방식 자체가 다르다**.

| | LeWM | EEG-JEPA (현재) |
|---|---|---|
| 예측 방식 | Auto-regressive (t→t+1) | Masked prediction (빈칸 채우기) |
| 인과성 | 인과적 (미래 못 봄) | 양방향 (앞뒤 모두 봄) |
| 마스킹 | 없음 | 60% 블록 마스킹 |
| Predictor | ARPredictor + AdaLN | EEGPredictor + mask token |
| Loss | 전체 시퀀스 MSE | 마스킹된 위치만 MSE |

### 우려
- SIGReg가 "마스킹 JEPA + stop-grad" 환경에서도 LeWM처럼 잘 작동하는지 실험적으로 미검증
- LeWM의 SIGReg weight(0.09)가 우리 설정에서도 적절한지 불확실

### 대응 계획
- Ablation Study에서 SIGReg weight를 조절하며 학습 안정성 확인
- stop-grad 유무에 따른 성능 비교 실험 설계
- 학습 초기에 collapse 징후(임베딩 분산 급락, loss 0 수렴) 모니터링

---

## Issue #5: 2-pass Encoding (Laya) vs 1-pass (현재 구현)

**상태: ⚠️ 검토 필요**

### 문제
Laya 논문에서는 Encoder를 **2번 통과**시킨다:
```
Pass 1: 전체 시퀀스 → Z (full embeddings) + z_cls (global summary) → StopGrad
Pass 2: 마스킹된 위치를 제외하고 → Z_ctx (context embeddings)
```

현재 우리 구현은 **1-pass**로 전체 시퀀스를 인코딩한 후 Predictor에서 마스킹을 처리한다.

### 영향
- 1-pass: 단순하고 빠르지만, context encoder가 마스킹된 위치 정보도 "보고" 있음
- 2-pass: Laya처럼 context가 정말로 관찰된 부분만 보게 되므로 더 정직한 예측

### 대응 계획
- 1차적으로 현재 1-pass 구현으로 학습 가능 여부 확인
- 성능이 부족하면 2-pass 방식으로 전환 검토

---

## Issue #6: 가변 채널 배칭 전략 — 채널 상한 vs 원본 유지

**상태: ✅ 해결됨 (2026-03-31)**

### 문제
REVE(92개 데이터셋) + TUH 혼합 학습 시 배치 내에 채널 수가 다른 샘플이 섞임 (19ch, 64ch, 128ch 등). PyTorch 배치 텐서는 같은 shape이어야 한다.

### 검토된 대안들
| 대안 | 기각 이유 |
|---|---|
| 채널 상한(64ch) + 랜덤 서브샘플링 | "모든 채널 수 대응" 주장과 모순. 채널 정보 손실 |
| Bucket Sampling (채널 수별 그룹 배칭) | 불필요한 복잡도. REVE의 82%가 19~22ch라 버켓 극심한 불균형 |

### 최종 해결: padding_mask 기반 원본 유지
- collate에서 배치 내 max_c까지 zero-padding + padding_mask 생성
- Channel Mixer의 cross-attention에 padding_mask 전달하여 가짜 채널 무시
- Channel Mixer(LUNA)가 원래 가변 채널 처리를 위해 설계된 구조이므로, 모델 철학과 완전히 일치

### 근거
- REVE 실제 채널 분포: 82%가 19~22ch → 대부분의 배치에서 패딩 2~3채널 수준
- 128ch 혼입 시 메모리 스파이크는 gradient accumulation 등 인프라 레벨에서 해결

---

## Issue #7: CAR과 Zero-padding의 상호작용

**상태: ✅ 해결됨 (2026-03-31)**

### 문제
CAR(Common Average Reference)을 collate 이후(패딩 후)에 적용하면, zero-padding된 가짜 채널이 평균 계산에 포함되어 실제 뇌파 신호가 왜곡된다.

```python
# 문제 상황
eeg = [19ch 실제 데이터 | 45ch 제로패딩]  # 64ch로 패딩됨
car = eeg - eeg.mean(dim=channel)  # ← 제로패딩이 평균을 왜곡!
```

### 해결: CAR 적용 순서 변경
- CAR을 **Dataset.__getitem__ 내**에서 적용 (패딩 전)
- 원본 채널만으로 평균 계산 → 이후 collate에서 패딩
- Mask-aware CAR 로직이 완전히 불필요해짐

---

## Issue #8: REVE 채널 분포 데이터 신뢰성

**상태: ✅ 해결됨 (2026-03-31)**

### 문제
외부 보고서에서 "REVE의 고밀도(65~128ch+) 데이터가 전체의 54.2%"라고 주장. 이를 기반으로 채널 상한 64ch + 서브샘플링 전략이 제안됨.

### 검증 결과 (REVE 공식 사이트 기반)
| 버켓 | 보고서 주장 | 실제 |
|---|---|---|
| ≤24ch | 5.7% | **82.4%** |
| 25~64ch | 40.1% | **2.9%** |
| 65~128ch+ | 54.2% | **14.7%** |

주요 구성: 19ch 41.0% (TUH), 21ch 35.2% (ICARE), 22ch 6.0%

### 영향
- 채널 상한 전략의 근거가 무효화됨
- 단, 최종 결론(원본 유지 + padding_mask)은 분포와 무관하게 유효
- **교훈: 외부 보고서의 수치는 반드시 원본 소스에서 교차 검증해야 함**

---

## Issue #9: 사전학습-벤치마크 데이터 누출 (Data Leakage)

**상태: ✅ 해결됨 (2026-03-31) — 방향 변경**

### 문제
REVE에 TUH가 포함(26,847시간, 44%)되어 있고, TUAB/TUSL은 TUH의 부분집합이다.

### 검토 과정
1. 초기: Laya 방식(피험자 단위 제거) 채택 시도
2. 발견: REVE 메타데이터에 피험자 ID가 없어 구현 불가
3. 재검토: SSL 사전학습은 라벨을 사용하지 않으므로, 대부분의 논문(LuMamba, BIOT 등)이 이를 leakage로 취급하지 않음

### 최종 해결: REVE 전체 사전학습 + 4종 다운스트림
- TUAB: 표준 벤치마크로 포함 (업계 관행과 동일)
- TDBrain/APAVA/BCI(MI): TUH와 완전 무관한 벤치마크 3종 추가
- 비TUH 벤치마크에서도 우수하면 → "leakage 없는 환경에서도 성능 검증"이 자동으로 성립

---

## Issue #10: TUH 별도 확보 불필요

**상태: ✅ 해결됨 (2026-03-31)**

### 문제
TUH 데이터 접근 승인이 어려움.

### 해결
REVE 논문 Appendix B 확인 결과, REVE에 TUH가 이미 포함 (14,987명, 26,847시간).
TUH를 별도로 받으면 오히려 중복 데이터 발생.

**→ REVE 단일 소스로 확정. TUH 별도 확보 불필요.**

---

## Issue #11: REVE 전처리 서버 한계 (OOM + 디스크 풀)

**상태: ✅ 해결됨 (2026-04-02) — 서버 이관 결정**

### 문제
학습 서버(422GB SSD, 62GB RAM)에서 REVE 전처리 3회 연속 실패:
1. `preprocess_parallel.py`: 디스크 100% → 업로드 deadlock
2. `preprocess_pipeline.py`: tmux 세션 종료 → 프로세스 같이 사망
3. `setsid nohup` 실행: OOM killer에 의해 사망 (55GB RSS)

### 원인
- Recording당 ~15GB raw data → 2개 동시 다운로드만으로 메모리 초과
- 전처리 산출물 + HF 캐시 + 업로드 대기 shard로 디스크 순식간에 포화

### 해결
- 81/322 recordings까지 처리된 shards (835개) HF에 업로드 완료
- 나머지 전처리는 4TB SSD 전처리 서버에서 수행 예정
- 학습 서버에서는 전처리 하지 않기로 결정

---

## Issue #12: HF 스트리밍 학습 병목 (GPU 가동률 15%)

**상태: ✅ 해결됨 (2026-04-02) — MosaicML Streaming 도입**

### 문제
HF datasets streaming으로 학습 시 GPU가 85% 시간 동안 유휴 상태:
- GPU 연산: 0.27s/step
- HF 데이터 로딩: 1.8s/step (avg), 최대 48s (shard 전환 시)
- GPU 가동률: 15%

### 검토된 대안들
| 대안 | 문제점 |
|---|---|
| 로컬 디스크 저장 | Full 데이터 ~2.5TB > 서버 422GB |
| 커스텀 더블버퍼 | 동작하나 race condition, 검증 부담 |
| 300GB 로컬 캐시 | GPU가 병목이라 캐시 늘려도 시간 동일 |

### 해결: MosaicML Streaming
- `hf://` 프로토콜로 HuggingFace 네이티브 지원
- `cache_limit` 설정으로 LRU 캐시 자동 관리
- 내장 prefetch, shuffle, multi-worker
- Parquet → MDS 포맷 변환 필요 (전처리 서버에서 수행 예정)
- 로컬 MDS 테스트: 0.234s/step (6.6x 빠름, GPU ~100%)

---

## Issue #13: TUAB 벤치마크 접근 불가

**상태: ⚠️ 보류 중**

### 문제
TUAB (Temple University Abnormal EEG Corpus) DUA 신청이 기각됨.
표준 벤치마크이나 현재 데이터 접근 불가.

### 대응
- TUAB 대신 접근 가능한 벤치마크로 먼저 진행:
  - BCI IV 2a (4-class MI, 공개, MOABB) — 구현 완료
  - APAVA (알츠하이머, 공개, OSF) — 구현 예정
- TDBrain (파킨슨, DUA 필요) — 별도 신청 필요
- TUAB 접근이 향후 가능해지면 추가

---

## Issue #14: 체크포인트 디스크 관리

**상태: ✅ 해결됨 (2026-04-02)**

### 문제
학습 서버 디스크가 422GB로, 여러 실험의 체크포인트를 로컬에 쌓으면 디스크 부족.
모델 19.5M × optimizer state 포함 = ~234MB/checkpoint.
10회 실험 × 20 checkpoints = ~47GB.

### 해결
- 체크포인트를 HF public model repo에 업로드 (`checkpoints/{run_name}/epoch_XXXX.pt`)
- 로컬에는 최신 3개만 유지 (older는 자동 삭제)
- 실험 완료 후 HF repo에서 best 모델만 남기고 정리
