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

## Research Context

Prior conversation history and paper analyses are stored in `EEG Preprocessing Methods in Research Papers.zip`. This contains markdown reports and paper excerpts covering: Brain-JEPA, EEG-VJEPA, S-JEPA, HEAR, LUNA, SPEED, DELTA, LaBraM, NeuroGPT, EEGPT, DeWave, Laya, LuMamba, and LeWM.
