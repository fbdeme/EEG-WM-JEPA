# TODO

> 연구 진행사항 관리. 완료 시 [x]로 체크하고 날짜를 기록합니다.

---

## 1. 연구 설계 (Research Design)

- [x] 선행 연구 서베이 및 분류 체계 수립 (2026-03-29)
- [x] 핵심 아키텍처 결정: LeWM(SIGReg) vs VQ-JEPA → Track A 확정 (2026-03-30)
- [x] 방법론 3단계 파이프라인 확정 (2026-03-30)
- [x] 전처리 표준화 5요소 정의 (2026-03-30)
- [ ] 하이퍼파라미터 설계 (패치 크기, 임베딩 차원, 쿼리 수 등)
- [x] 사전학습 데이터셋 선정: REVE(4.4TB) + TUH(1.6TB) (2026-03-31)
- [x] 데이터 파이프라인 설계 확정 (2026-03-31)
- [ ] 연구 제안서 최종 작성

## 2. 구현 - Stage 1: EEG 표준화

- [x] 프로젝트 구조 및 환경 설정 (uv, PyTorch, MNE) (2026-03-30)
- [x] 리샘플링 모듈 구현 (→ 200Hz) (2026-03-30)
- [x] Common Average Re-referencing 구현 (2026-03-30)
- [x] Bandpass 필터 구현 (0.5~75Hz) (2026-03-30)
- [x] 2초 윈도우 세그멘테이션 구현 (2026-03-30)
- [x] 더미 데이터셋 로더 구현 (2026-03-30)
- [ ] 채널 Alias 매핑 로직 구현 (REVE Position Bank 연동)
- [ ] 드롭 비율 로깅 및 품질 통제 로직
- [ ] 오프라인 전처리 스크립트 (.edf → LitData)
- [ ] Custom collate_fn 구현 (가변 채널 padding + padding_mask)
- [ ] Channel Mixer에 padding_mask 지원 추가

## 3. 구현 - Stage 2: 사전학습

- [x] Patch Embedder 구현 (depthwise 1D Conv) (2026-03-30)
- [x] Dynamic Channel Mixer 구현 (Fourier encoding + cross-attention) (2026-03-30)
- [x] Spatiotemporal Masking 전략 구현 (2026-03-30)
- [x] Transformer Encoder 구현 — LeWM 코드 기반 (2026-03-30)
- [x] Predictor 구현 (2026-03-30)
- [x] Loss 함수 구현 (Prediction + SIGReg + Query Specialization) (2026-03-30)
- [x] 더미 데이터 forward/backward pass 검증 (2026-03-30)
- [ ] LitData StreamingDataset 통합
- [ ] 학습 루프 및 로깅 구현
- [ ] 사전학습 실행 및 수렴 확인

## 4. 구현 - Stage 3: 다운스트림 평가

- [ ] Linear Probing 평가 파이프라인
- [ ] Fine-tuning 평가 파이프라인
- [ ] TUAB 벤치마크 실험
- [ ] TUSL 벤치마크 실험
- [ ] BCI Competition 벤치마크 실험
- [ ] TDBrain 벤치마크 실험
- [ ] Ablation Study 설계 및 실행

## 5. 논문 작성

- [ ] Introduction 작성
- [ ] Related Work 작성
- [ ] Method 작성
- [ ] Experiments & Results 작성
- [ ] Discussion & Conclusion 작성
- [ ] 그림/표 제작
- [ ] 투고 대상 학회/저널 선정
