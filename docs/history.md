# Research History

> 방법론의 변경 이력과 연구 과정에서의 주요 의사결정을 기록합니다.

---

## 2026-03-29 ~ 03-30: 초기 연구 설계

### Phase 1: EEG 전처리 다양성 탐색
- 기존 EEG 파운데이션 모델들(EEG-VJEPA, LaBraM, NeuroGPT, EEGPT, DeWave)의 전처리 방식 조사
- **핵심 발견**: EEG 데이터의 이질성(채널, 샘플링 레이트, 참조 전극, 노이즈, 세그멘테이션)이 파운데이션 모델의 근본적 병목

### Phase 2: JEPA 문헌 서베이
- 분석한 논문: Brain-JEPA(fMRI), EEG-VJEPA, S-JEPA, HEAR, LUNA, SPEED
- S-JEPA: 공간 블록 마스킹 도입 (최초 EEG-JEPA)
- HEAR: 3D 좌표 기반 전극 인코딩
- LUNA: Learned-query cross-attention으로 채널 통합 (채널 수 무관)

### Phase 3: 5축 분류 체계 수립
- EEG 파운데이션 모델 분류 체계 정립:
  1. 입력 표현 (비디오/텍스트/이산 토큰/독립 센서)
  2. 토폴로지 불변 전략 (교집합/ID 임베딩/좌표/learned-query)
  3. 학습 패러다임 (MAE/JEPA/autoregressive)
  4. 신호 정제 수준 (최소/주파수 분해/무거운 파이프라인)
  5. 다운스트림 태스크 (분류/텍스트 생성/비디오 생성)
- "EEG-JEPA" 연구 비전 확립: 3D 좌표 + 주파수 분해 → JEPA → World Model

### Phase 4: 경쟁 논문 대응
- **LuMamba, Laya** (2026.03) 발견 — 이미 "채널 통합 + JEPA" 구현
- 포기 대신 3가지 차별화 전략 수립:
  1. DELTA 스타일 주파수 대역 분해 추가
  2. 시공간 동시 마스킹 (temporal + spatial)
  3. EEG-to-Video World Model (장기 목표)

### Phase 5: 아키텍처 최종 결정
- **VQ-JEPA 아이디어 검토**: DELTA의 RVQ + JEPA 결합
  - 장점: 노이즈 필터링, 계층적 주파수 학습, LLM/Diffusion 호환
  - 문제: SIGReg(연속적 가우시안)과 RVQ(이산적 코드북)가 수학적으로 비호환
- **LeWM 발견**: LeCun팀의 최신 World Model, SIGReg 기반, EMA 완전 제거
- **최종 결정: Track A (LeWM 순정)**
  - RVQ 포기, SIGReg 채택 (학습 안정성 + 빠른 논문화)
  - Channel Mixer + Spatiotemporal Masking 결합이 핵심 기여
- **전처리 표준화 5요소 확정**: 채널, 샘플링 레이트, 참조 전극, 노이즈, 세그멘테이션
- **윈도우 크기 2초로 확정**: Laya/LuMamba와 공정 비교를 위해
- **방법론 3단계 구조 확정**: EEG 표준화 → LeWM 사전학습 → 다운스트림 평가

---

## 2026-03-31: 데이터 파이프라인 설계

### Phase 6: 데이터 파이프라인 설계 논의
- **REVE(4.4TB) + TUH(1.6TB) 결합** 결정 — "이질적 데이터 통합"이라는 연구 가설 증명에 최적
- TUH는 라이선스 제약 → 로컬 전처리 후 HuggingFace Private 업로드 전략 채택
- **LitData 도입** — 500GB SSD로 6TB 스트리밍 가능 (Prefetching 기반)
- 스트리밍 엔진 검토: 단순 HF streaming은 GPU Starvation 유발 → LitData로 해결

### Phase 7: 데이터 파이프라인 기술적 이슈 해결
- **채널 Alias 문제**: REVE Position Bank (543개 전극) 활용으로 해결. 드롭 비율 로깅 추가
- **CAR 적용 시점**: 패딩과의 상호작용 문제 발견 → Dataset 내 적용(패딩 전)으로 확정
- **가변 채널 배칭**: 채널 상한+서브샘플링, Bucket Sampling 등 검토 후 모두 기각
- **최종 결론**: "채널을 인위적으로 조작하지 않는다" — padding_mask 기반으로 원본 그대로 처리
- **REVE 채널 분포 검증**: 보고서의 "고밀도 54.2%" 주장이 실제로는 14.7%임을 확인 (실제 82%가 19~22ch)

### Channel Mixer 계보 확인
- LUNA (2025.10): Learned-query Cross-attention 최초 제안
- Laya (2026.03): LUNA의 Channel Mixer를 LeJEPA에 결합
- LuMamba (2026.03): LUNA의 Channel Mixer를 Mamba에 결합
- 우리의 추가 기여: padding_mask 지원 (기존 논문들은 동일 채널 수 데이터셋 내 학습만 수행)

### Phase 8: 사전학습 데이터 전략 확정
- **REVE에 TUH가 이미 포함** (26,847시간, 전체 44%) 확인 → TUH 별도 확보 불필요
- **데이터 누출 문제 발견**: TUAB/TUSL은 TUH의 부분집합 → REVE로 사전학습 시 leakage 위험
- 기존 연구 분리 전략 조사: Laya(피험자 단위, ★★★), LaBraM(TUH 미사용, ★★★), LuMamba/REVE(모호, ★★☆), NeuroGPT(문제 지적됨, ★☆☆)
- **Laya 방식 채택**: 벤치마크 test/val set 피험자의 모든 녹화를 REVE에서 제거
- 리뷰 논문(arXiv 2507.11783)이 TUH 사전학습+TUAB 평가의 체계적 문제 지적 → 우리는 이를 사전 방어
- **전처리 파이프라인 확정**: REVE에서 스트리밍 다운로드 → 전처리 → HF Private 업로드 → 원본 삭제 (디스크에 안 쌓임)

---

## 2026-04-01: 하이퍼파라미터 설계 및 로깅 구축

### Phase 9: 레퍼런스 논문 하이퍼파라미터 조사
- LeWM, Laya, LUNA, LaBraM, Brain-JEPA, S-JEPA, LuMamba 7개 논문의 하이퍼파라미터 비교 분석
- **LeWM의 독특한 설계 발견**: Predictor가 Encoder보다 2배 큰 구조 (Pred/Enc ≈ 2.0). 다른 JEPA(I-JEPA, Brain-JEPA)와 완전히 반대. SIGReg 방식에서 predictor 용량이 representation 품질에 직결
- **LeWM predictor의 dim_head 메커니즘 분석**: working dim은 encoder와 동일(192)하지만, attention 내부에서 `heads(16) × dim_head(64) = 1024`로 확장. 이 내부 확장이 파라미터의 대부분을 차지
- 기존 코드의 문제 발견: `dim_head = encoder_dim / heads`로 계산하여 heads를 늘려도 파라미터가 변하지 않음 → `predictor_dim_head`를 64로 고정하는 config 옵션 추가

### Phase 10: 모델 구조 하이퍼파라미터 확정
- **Encoder**: D256, L6→8, H4 (LUNA-Base 수준)
- **Predictor**: depth 3→6, heads 4→12, dim_head=64 고정, mlp=2048 (LeWM 비율 반영, Pred/Enc=1.75)
- **Patch size**: 40→25 (125ms, 16패치. Laya 기준, alpha 리듬 1주기 커버)
- **Channel Mixer**: mixer_dim 256→64 (4x 공간 압축 병목), queries=8 유지
- GPU 메모리 프로파일 실측: 19.5M 모델이 batch=1024, 128ch에서도 7.3GB만 사용 (A6000 48GB 대비 여유)
- 학습 파라미터도 레퍼런스 대비 조정: lambda_sigreg 1.0→0.09, lambda_query 0.1→0.8, sigreg_projections 64→512, batch_size 64→512

### Phase 10.5: 학습 파라미터 확정
- **예측 방식 논의**: LeWM의 autoregressive(미래 프레임 예측) vs masking(가려진 패치 예측) 비교 분석. LeWM의 AR은 프레임 간 예측이라 우리 파이프라인(독립 2초 윈도우)과 비호환. spatiotemporal masking이 논문 핵심 기여이므로 **masking 유지 확정**
- **bf16 전환**: fp16 + GradScaler → bf16 (GradScaler 제거). 레퍼런스 전부 bf16 사용, SIGReg의 분포 통계 계산에서 fp16 overflow 위험 제거
- **Warmup 10 epochs 확정**: 기존 step 기반(~2 epochs)에서 epoch 기반으로 변경. LUNA=10/60(17%), LaBraM=5/50(10%) 참고, 10%인 10 epochs 채택. SIGReg 초반 불안정 방지
- **Mask block 크기 유지 확정**: 0.5~1.0초(4~8패치). spatiotemporal 마스킹에서 시간 블록이 커지면 공간 마스킹 여지 감소. Laya(5~10패치)와 겹치는 범위

### Phase 11: wandb 로깅 체계 구축
- wandb 연동 완료. 매 step: loss 분해(pred/sigreg/query), grad_norm, lr, 배치 채널 통계. 매 epoch: 요약 + duration. 체크포인트: 5 epoch마다 wandb Artifact 업로드
- `.env`에 HF_TOKEN, WANDB_API_KEY 관리 (`.gitignore` 반영 완료)

---

## 2026-04-02: REVE 전처리 진행 및 서버 이관 결정

### Phase 12: REVE 전처리 실행 (81/322)
- `preprocess_parallel.py` (4 workers, idx 55→322): 디스크 100% → 업로드 deadlock으로 중단
- `preprocess_pipeline.py` (DL 2 + Proc 4 + Upload 1 분리): tmux 세션 종료 시 프로세스 같이 사망
- `setsid nohup`으로 완전 분리 실행: OOM killer에 의해 사망 (55GB RSS, 62GB 서버)
- **결론**: 422GB SSD + 62GB RAM 서버에서는 안정적 전처리 불가능
  - 원인 1: REVE recording당 ~15GB raw data, 2개 동시 다운로드만으로 메모리 초과
  - 원인 2: 전처리 산출물 + HF 캐시 + 업로드 대기 shard로 디스크 순식간에 포화
- **HF repo 현황** (`fbdeme/reve-preprocessed`): 835 shards 업로드 완료
  - idx 0-67: 완료 (644 shards)
  - idx 68-80: 유실 (디스크 풀 상태에서 로컬 shard 삭제됨, 재처리 필요)
  - idx 81-93: 완료 (191 shards)
- **서버 이관 결정**: 4TB SSD 서버에서 idx 68-80 + idx 94-322 재개 예정

---

## 변경 로그 (Method Changes)

| 날짜 | 변경 내용 | 이유 | 영향 |
|---|---|---|---|
| 2026-03-30 | RVQ(VQ-JEPA) 기각 → SIGReg(LeWM) 채택 | SIGReg와 RVQ의 수학적 비호환 | 잠재 공간이 연속적으로 고정 |
| 2026-03-30 | 윈도우 크기 2초 확정 | Laya/LuMamba baseline과 공정 비교 | 전처리 파이프라인 확정 |
| 2026-03-30 | 방법론 3단계 구조 확정 | 논문 구조와 매핑 | research_method.md v1.0 |
| 2026-03-31 | 채널 상한/서브샘플링 기각 | 모델의 Topology-Invariant 주장과 모순 | 원본 채널 그대로 유지 |
| 2026-03-31 | CAR 적용 시점: Dataset 내 (패딩 전) | 패딩 채널이 CAR 평균 왜곡 방지 | Mask-aware 로직 불필요 |
| 2026-03-31 | 데이터 파이프라인 v1.0 확정 | REVE+TUH 혼합, LitData, padding_mask | data_pipeline.md 신규 |
| 2026-03-31 | research_method.md v1.1 업데이트 | 데이터 파이프라인 반영 | 전처리 섹션 갱신 |
| 2026-03-31 | TUH 별도 확보 기각 → REVE 단일 소스 | REVE에 TUH 포함 (44%), 중복 방지 | data_pipeline.md v2.0 |
| 2026-03-31 | 피험자 단위 데이터 분리(Laya 방식) 채택 | Leakage 방지, 리뷰어 공격 방어 | 전처리에 필터링 단계 추가 |
| 2026-03-31 | 피험자 단위 제거 기각 → REVE 전체 사전학습 확정 | REVE 메타데이터에 피험자 ID 없음. SSL은 라벨 미사용이므로 leakage 논쟁 최소. 비TUH 벤치마크 추가로 방어 | data_pipeline.md v3.0 |
| 2026-03-31 | 다운스트림 4종 확정 (TUAB+TDBrain+APAVA+MI) | TUAB로 표준 비교 + 비TUH 3종으로 전이 능력 증명 | research_method.md 벤치마크 갱신 |
| 2026-04-01 | predictor_dim_head=64 고정 옵션 추가 | LeWM 분석 결과: dim_head 고정이 핵심 | eeg_jepa.py 수정 |
| 2026-04-01 | 모델 구조 하이퍼파라미터 확정 (19.5M) | LeWM/Laya/LUNA 비율 분석 기반 | configs/default.yaml 갱신 |
| 2026-04-01 | 학습 파라미터 조정 (SIGReg, query, batch) | 레퍼런스 대비 불일치 수정 | configs/default.yaml 갱신 |
| 2026-04-01 | wandb 로깅 체계 구축 | 학습 분석 및 모니터링 | main.py 갱신, wandb 연동 |
| 2026-04-01 | Masking 예측 방식 유지 확정 (AR 기각) | spatiotemporal masking이 논문 핵심 기여, AR은 파이프라인 비호환 | 구조 변경 없음 |
| 2026-04-01 | fp16 → bf16 전환, GradScaler 제거 | 레퍼런스 전부 bf16, SIGReg overflow 방지 | main.py 갱신 |
| 2026-04-01 | Warmup step 기반 → 10 epochs 기반 | LUNA/LaBraM 참고, ~10% 표준 | main.py + config 갱신 |
| 2026-04-01 | Mask block 0.5~1.0초 유지 확정 | spatiotemporal 마스킹과의 균형 | 변경 없음 |
| 2026-04-01 | 논문 Introduction + Related Work 초안 | 40+ 논문 인용 검증 포함 | paper/main.md |
| 2026-04-02 | REVE 전처리 81/322 완료, 서버 이관 결정 | 422GB 서버에서 OOM+디스크 부족 3회 실패 | 4TB SSD 서버에서 재개 예정 |
| 2026-04-02 | wandb 로깅 확장 (SIGReg 내부통계, 컴포넌트별 grad, throughput, GPU 메모리) | 하이퍼파라미터 분석 커버리지 강화 | eeg_jepa.py + main.py 갱신 |
| 2026-04-02 | 체크포인트 HF public repo 업로드 + 로컬 3개 유지 | 서버 디스크 절약, 실험 간 모델 공유 | main.py 갱신 |
| 2026-04-02 | HF 스트리밍 사전학습 파이프라인 검증 완료 | 835 shards로 5 step 테스트: loss 72→36, 정상 수렴 | 전처리 완료 후 바로 학습 가능 |
