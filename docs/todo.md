# TODO

> 연구 진행사항 관리. 완료 시 [x]로 체크하고 날짜를 기록합니다.

---

## 1. 연구 설계 (Research Design)

- [x] 선행 연구 서베이 및 분류 체계 수립 (2026-03-29)
- [x] 핵심 아키텍처 결정: LeWM(SIGReg) vs VQ-JEPA → Track A 확정 (2026-03-30)
- [x] 방법론 3단계 파이프라인 확정 (2026-03-30)
- [x] 전처리 표준화 5요소 정의 (2026-03-30)
- [x] 하이퍼파라미터 설계 — 모델 구조 확정 (2026-04-01)
  - [x] 레퍼런스 논문(LeWM/Laya/LUNA/LaBraM/Brain-JEPA) 하이퍼파라미터 비교 조사
  - [x] Encoder 확정: D256, L8, H4 (LUNA-Base 수준)
  - [x] Predictor 확정: L6, H12, dim_head=64 (LeWM 비율 Pred/Enc=1.75)
  - [x] Patch size 확정: 25 (125ms, 16패치)
  - [x] Channel Mixer 확정: mixer_dim=64, queries=8 (4x 공간 압축 병목)
  - [x] GPU 메모리 프로파일 실측 (19.5M, batch=1024 128ch에서 7.3GB)
  - [x] 학습 파라미터 확정: bf16, warmup 10 epochs, mask_block 유지 (2026-04-01)
  - [x] 예측 방식 확정: masking 유지 (autoregressive 기각) (2026-04-01)
  - [ ] 학습 시 검증: grad_accum, SIGReg projections 512 vs 1024
- [x] 사전학습 데이터셋 선정: REVE(4.4TB) + TUH(1.6TB) (2026-03-31)
- [x] 데이터 파이프라인 설계 확정 (2026-03-31)
- [x] hyperparameter_design.md 문서화 (2026-04-01)
- [ ] 연구 제안서 최종 작성

## 2. 구현 - Stage 1: EEG 표준화

- [x] 프로젝트 구조 및 환경 설정 (uv, PyTorch, MNE) (2026-03-30)
- [x] 리샘플링 모듈 구현 (→ 200Hz) (2026-03-30)
- [x] Common Average Re-referencing 구현 (2026-03-30)
- [x] Bandpass 필터 구현 (0.5~75Hz) (2026-03-30)
- [x] 2초 윈도우 세그멘테이션 구현 (2026-03-30)
- [x] 더미 데이터셋 로더 구현 (2026-03-30)
- [x] 채널 Alias 매핑 로직 구현 (REVE Position Bank 연동) (2026-03-31)
- [x] REVE 데이터 로더 구현 (raw memmap + 메타데이터 CSV) (2026-03-31)
- [ ] 드롭 비율 로깅 및 품질 통제 로직
- [x] 오프라인 전처리 스크립트 (REVE .npy → HF parquet) (2026-03-31)
- [x] Custom collate_fn 구현 (가변 채널 padding + padding_mask) (2026-03-31)
- [x] Channel Mixer에 padding_mask 지원 추가 (2026-03-31)

## 3. 구현 - Stage 2: 사전학습

- [x] Patch Embedder 구현 (depthwise 1D Conv) (2026-03-30)
- [x] Dynamic Channel Mixer 구현 (Fourier encoding + cross-attention) (2026-03-30)
- [x] Spatiotemporal Masking 전략 구현 (2026-03-30)
- [x] Transformer Encoder 구현 — LeWM 코드 기반 (2026-03-30)
- [x] Predictor 구현 (2026-03-30)
- [x] Loss 함수 구현 (Prediction + SIGReg + Query Specialization) (2026-03-30)
- [x] 더미 데이터 forward/backward pass 검증 (2026-03-30)
- [x] HF Streaming Dataset 통합 (LitData → HF datasets streaming으로 변경) (2026-03-31)
- [x] 학습 루프 및 로깅 구현 (AMP, grad accum, cosine scheduler) (2026-03-31)
- [x] wandb 로깅 체계 구축 (step/epoch 로깅, artifact 업로드) (2026-04-01)
- [x] wandb 로깅 확장: SIGReg 내부통계, 컴포넌트별 grad norm, throughput, GPU 메모리 (2026-04-02)
- [x] 체크포인트 HF public repo 업로드 + 로컬 최신 3개만 유지 (2026-04-02)
- [x] HF 스트리밍 사전학습 파이프라인 검증 (835 shards, loss 수렴 확인) (2026-04-02)
- [ ] REVE 전처리 실행 (322개 레코딩 → HF Private repo 업로드) — 진행 중 (81/322 완료, 835 shards on HF) (2026-04-02)
  - idx 0-67: 완료 (644 shards)
  - idx 68-80: 미완료 (디스크 풀로 유실, 재처리 필요)
  - idx 81-93: 완료 (191 shards)
  - idx 94-322: 미처리 (228 recordings 남음, ~3.6TB raw)
  - ⚠️ 현재 서버(422GB SSD)에서 OOM + 디스크 부족으로 3회 실패 → 4TB SSD 서버로 이관 예정
- [ ] 사전학습 실행 및 수렴 확인

## 4. 구현 - Stage 3: 다운스트림 평가

- [ ] Linear Probing 평가 파이프라인
- [ ] Fine-tuning 평가 파이프라인
- [ ] ① TUAB 벤치마크 (정상/비정상, 표준 비교)
- [ ] ② TDBrain 벤치마크 (파킨슨, 26ch, TUH 무관)
- [ ] ③ APAVA 벤치마크 (알츠하이머, TUH 무관)
- [ ] ④ BCI/MI 벤치마크 (운동 상상, TUH 무관)
- [ ] Ablation Study 설계 및 실행

## 5. 논문 작성

- [x] 논문 초안 구조 잡기 (paper/main.md) (2026-04-01)
- [x] Introduction 작성 (검증된 인용 포함) (2026-04-01)
- [x] Related Work 작성 (4개 서브섹션, 비교 테이블) (2026-04-01)
- [x] References 정리 (40+ 논문 검증) (2026-04-01)
- [ ] Method 작성 (하이퍼파라미터 확정 내용 반영)
- [ ] Experiments & Results 작성 (학습 완료 후)
- [ ] Ablation Studies 실행 및 작성
- [ ] Discussion & Conclusion 작성
- [ ] 그림/표 제작 (아키텍처 다이어그램, 결과 테이블)
- [ ] 투고 대상 학회/저널 선정
