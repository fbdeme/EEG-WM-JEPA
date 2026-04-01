# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG-WM-JEPA is a research project building an EEG foundation model that combines:
- **LeWM (LeWorldModel)** — JEPA-based world model using SIGReg regularization (no EMA/stop-gradient)
- **LUNA-style Channel Mixer** — Learned-query cross-attention for topology-invariant EEG processing
- **Spatiotemporal Masking** — Simultaneous time + spatial masking for self-supervised pretraining

The goal is a universal EEG encoder that works across any electrode montage/equipment, pretrained via self-supervised learning, evaluated on classification benchmarks (TUAB, TUSL, BCI, TDBrain).

## Architecture (3-Stage Pipeline)

### Stage 1: EEG Standardization
- Resample → 200Hz
- Re-reference → Common Average Reference
- Bandpass filter → 0.5–75Hz
- Segment → 2-second windows

### Stage 2: Self-Supervised Pretraining
```
Raw EEG [B, C, T] (C is variable per dataset)
  → Patch Embedder (depthwise 1D Conv, channel-independent)
  → Channel Mixer (3D electrode coords + Fourier encoding + cross-attention → fixed-size [B, N, D])
  → Spatiotemporal Masking (60%, 500ms–1s blocks)
  → Transformer Encoder (RoPE positional embeddings)
  → Predictor (predicts masked region latent vectors)
  → Loss: Prediction Loss (MSE) + SIGReg + Query Specialization Loss
```

### Stage 3: Downstream Evaluation
- Linear probing (encoder frozen) and fine-tuning (encoder trainable)
- Benchmarks: TUAB (normal/abnormal), TUSL (sleep staging), BCI Competition (motor imagery), TDBrain

## Key Technical Decisions

- **SIGReg over RVQ**: Continuous latent space (SIGReg) chosen over discrete tokenization (RVQ) for training stability. These two approaches are mathematically incompatible.
- **Minimal preprocessing**: No ICA or heavy artifact removal. JEPA's latent prediction naturally filters noise.
- **2-second windows**: Matches Laya/LuMamba for fair baseline comparison.
- **Channel Mixer from LUNA/Laya**: Handles variable channel counts (8–256) by projecting to fixed-size representation via learned queries.

## Reference Codebases

- LeWM: https://github.com/lucas-maes/le-wm (SIGReg + Predictor architecture)
- Laya: LeJEPA-based EEG foundation model (Channel Mixer + Patch Embedder reference)
- LuMamba: LeJEPA + Mamba backbone for EEG
- DELTA: RVQ tokenizer for EEG (not used, but studied for comparison)

## Documentation Convention

작업 완료 후 반드시 `docs/` 내 관련 문서를 업데이트할 것:

- **`docs/todo.md`**: 완료된 항목 `[x]` 체크 + 날짜 기록. 진행 중인 항목은 상태 표시 (예: "— 진행 중 (54/322)")
- **`docs/history.md`**: 주요 의사결정이나 기술적 발견이 있을 때 Phase 번호로 추가. 변경 로그 테이블에도 한 줄 추가
- **`docs/data_pipeline.md`**: 데이터 파이프라인 변경 시
- **`docs/hyperparameter_design.md`**: 하이퍼파라미터 변경 시
- **`configs/default.yaml`**: 하이퍼파라미터 값 변경 시 config에 반영

코드만 고치고 문서를 안 고치면 안 됨. 사용자가 별도로 요청하지 않아도 관련 docs는 함께 업데이트할 것.

## Research Context

Prior conversation history and paper analyses are stored in `EEG Preprocessing Methods in Research Papers.zip`. This contains markdown reports and paper excerpts covering: Brain-JEPA, EEG-VJEPA, S-JEPA, HEAR, LUNA, SPEED, DELTA, LaBraM, NeuroGPT, EEGPT, DeWave, Laya, LuMamba, and LeWM.
