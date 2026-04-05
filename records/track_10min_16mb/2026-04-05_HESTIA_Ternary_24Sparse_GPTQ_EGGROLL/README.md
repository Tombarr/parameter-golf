# HESTIA Ternary: Experiments in Differentiable QAT, 2:4 Packing, and GPTQ for Ternary

**Status: Experimental — novel techniques validated but not yet competitive**

Fork of [Ciprian-Florin Ifrim's ternary submission](../2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/) (1.1570 BPB) exploring four novel techniques. Honest write-up of what worked, what didn't, and why.

---

## H100 Results (1×H100 SXM, sp1024 vocab)

### Best Run: STE Ternary Baseline (MLP 8×, 11L, 124.7M params)

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | **1.3813** |
| Roundtrip val_bpb | **1.4525** (gap: +0.071) |
| Sliding val_bpb | **1.4285** (stride=16, T=1.05) |
| Artifact | **27.04 MB — OVER 16MB BUDGET** |
| Training steps | 1300 in 541s (418ms/step) |
| Params | 124,748,632 (11L × 768d × MLP 8×) |

The model trains well but **124M params don't fit in 16MB**. Real compression ratio for properly-trained ternary weights is **~1.76 bits/weight** with base3+LZMA. Maximum ternary params for 16MB: **~72M**.

### All H100 Runs

| Run | Config | Steps | Loss | BPB | Outcome |
|-----|--------|-------|------|-----|---------|
| hestia_1h100 | HESTIA on, MLP4, 10L | 300 | 2.75 | — | **NaN at step 400** (tau too aggressive) |
| hestia_slow_anneal | HESTIA tau_init=0.5, ramp=0.6, MLP6, 10L | 350 | 2.77 | — | **NaN at step 400** (same root cause) |
| hestia_taufix2_11L | HESTIA tau clamp=0.15, MLP6, 11L | 700 | 2.47 | — | **NaN at step 800** (tau=0.22, still unstable) |
| baseline_ste_11L | STE, GPTQ on, MLP6, 11L | 1480 | 2.27 | 1.37 | GPTQ corrupted roundtrip (15.71 BPB) |
| **clean_ste_11L** | **STE, clean, MLP8, 11L** | **1300** | **2.29** | **1.38** | **Stable but 27MB over budget** |

---

## What Worked

### Standard STE Ternary Training
The proven STE approach from the original ternary submission trains stably on H100 with Muon optimizer, FA3, FP8 storage, and torch.compile. Loss curves are smooth and consistent across all runs.

### FA3/SDPA Auto-Fallback
Runtime detection of GPU compute capability (≥9.0 for FA3, SDPA fallback otherwise). Enables the same script to run on RTX 3090, H100, or CPU without changes.

### Vectorized 2:4 Pack/Unpack
Numpy-vectorized packing/unpacking replacing Python for-loops. ~100× faster serialization for the 2:4 path.

### LeakyReLU(0.5)² Activation
Adopted from SOTA submissions. Free -0.003 BPB over relu² at zero compute cost.

### 2048 Sequence Training with YaRN
Training at 2048 tokens (vs 1024) with YaRN positional encoding. Uses more VRAM but teaches longer-range dependencies directly.

---

## What Didn't Work

### 1. HESTIA Differentiable QAT — Unstable with Muon

**Source**: [HESTIA: Hessian-Guided Differentiable QAT](https://arxiv.org/abs/2601.20745) (Wang et al., ICML 2026)

HESTIA replaces STE with a differentiable softmax relaxation over {-1, 0, 1}, temperature-annealed from soft to hard. In theory, this provides true gradients instead of the STE approximation.

**Result**: NaN loss in every run, consistently around step 400-800. The failure mode is the interaction between HESTIA's softmax gradients and Muon's Newton-Schulz orthogonalization — the gradient magnitudes amplify each other during the temperature annealing phase.

Attempted mitigations (all failed):
- Slower pressure ramp (0.2 → 0.6 of training): NaN at step 400
- Higher tau_init (0.3 → 0.5): NaN at step 400
- Lower sensitivity alpha (0.4 → 0.2): NaN at step 800
- tau_eff floor clamped to 0.15: NaN at step 800 with tau=0.22

**Potential fixes (untested)**:
- Use Adam instead of Muon for ternary layers
- Gradient clipping (GRAD_CLIP_NORM=1.0)
- Much higher tau floor (0.25+)
- Lower MATRIX_LR (0.02) when HESTIA active

**Conclusion**: HESTIA was designed for AdamW at 1B+ scale with 10B tokens. Adapting it for Muon at 70-120M scale with <200M tokens requires significant hyperparameter work. The original HESTIA paper did not test with Muon.

### 2. GPTQ Error Compensation for Ternary — Corrupts Weights

Adapted Full Hessian GPTQ for the ternary grid with per-group scale matching.

**Result**: Roundtrip BPB exploded from 1.37 to 9.46 (pre-quant to post-quant). The column-by-column error accumulation diverges on ternary weights, where the quantization grid {-1, 0, 1} is too coarse for the error redistribution to converge.

Additional issue: GPTQ calibration took **344 seconds** (64×2048 autoregressive generation + column processing of 44 layers), blowing the time budget.

**Root cause investigation**: The GPTQ-corrupted weights happened to compress extremely well (2.57 MB for 97M params) because the divergent error accumulation produced degenerate, highly-repetitive ternary patterns. This created a false signal that 97M params fit in 16MB.

**Conclusion**: Standard GPTQ assumes the quantization error per column is small relative to the weight magnitude. For ternary (3 values), the per-column error is massive, and redistribution amplifies rather than compensates it. GPTQ for ternary likely needs a completely different approach — possibly block-wise optimization (as in EGGROLL) rather than column-wise error propagation.

### 3. 2:4 Structured Ternary Packing — Worse Than Base3+LZMA

Constrains every group of 4 weights to have exactly 2 nonzeros, enabling deterministic 1.25 bits/weight encoding.

**Result**: 2:4 packed artifact was **4× larger** than base3+LZMA (10.52 vs 2.57 MB on the GPTQ-corrupted run, ~similar ratio expected on clean weights). LZMA's adaptive dictionary compression exploits the natural zero patterns in standard ternary (~34% zeros) far more effectively than the structured encoding.

**Why**: The 2:4 encoding removes the long zero-runs and sign-pattern repetition that LZMA exploits. By forcing a fixed structure, we destroy the compressibility that makes base3+LZMA efficient. The theoretical 1.25 bits/weight is before LZMA; after LZMA, standard base3 achieves ~1.76 bits/weight which is only 0.51 bits more, while the 2:4 structure prevents LZMA from achieving similar gains.

**Conclusion**: Custom packing schemes must be evaluated AFTER general-purpose compression, not before. LZMA is very good at exploiting natural statistical patterns in ternary weights.

### 4. Artifact Size — 72M Param Ceiling

With properly-trained ternary weights (not GPTQ-corrupted), base3+LZMA achieves **~1.76 bits/weight**. This means:

```
16 MB × 8 bits / 1.76 bits = ~72M max ternary params
```

This matches the original ternary submission (73.7M). Our attempts at 88-124M params exceeded the budget. The original authors already found the optimal model size for this compression format.

---

## Artifact Budget Reality

| Params | MLP | Layers | Base3+LZMA | Fits? |
|--------|-----|--------|-----------|-------|
| 66M | 4× | 10 | ~14.5 MB | ✓ |
| 73M | 4× | 10 (original) | ~16.0 MB | ✓ (tight) |
| 88M | 6× | 10 | ~19.4 MB | ✗ |
| 98M | 6× | 11 | ~21.5 MB | ✗ |
| 124M | 8× | 11 | ~27.0 MB | ✗ |

The 2.57 MB artifact in the GPTQ run was an artifact of weight corruption, not real compression.

---

## Bugs Found & Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| GPTQ scale mismatch | Per-row vs per-group scale in quant/dequant | GPTQ uses per-group scales with column-group indexing |
| base3 deq corruption | `scale/q_absmean` correction wrong for GPTQ | Removed, use `q * scale` directly |
| 2:4 random zeroing | topk on ternary (all magnitude 1) picked randomly | Mask from continuous weights before quantization |
| FA3 crash on non-Hopper | FA3 installed but runtime-incompatible | Check compute capability ≥ 9.0, SDPA fallback |
| `generate()` dtype | float32 passed to FA3 (needs bf16) | Wrapped in autocast |
| `.item()` graph break | torch.compile recompiles on scalar guards | `torch.where` + tensor buffers |
| Hutch++ reshape crash | `8*seq_len+1` ≠ `8*(seq_len+1)` | Fixed token count formula |
| uint16 token indexing | Shard tokens are uint16, ops need int64 | Cast to `.long()` |
| DDP + torch.compile + HESTIA | autograd hook assertion with graph breaks | Use plain python3 for 1-GPU (DDP fix still needed for 8-GPU) |
| HESTIA NaN | tau < 0.1 causes gradient explosion with Muon | Clamp tau_eff min=0.15 (insufficient — root cause is Muon interaction) |

---

## Test Coverage

40 tests across 3 suites, all passing:

| Suite | Tests | Coverage |
|-------|-------|----------|
| `test_components.py` | 5 | HESTIA math, 2:4 packing, GPTQ, EGGROLL, Hutch++ |
| `test_cuda_script.py` | 20 | Real script components: model forward/backward, serialization |
| `test_regressions.py` | 15 | Every bug found during GPU validation |

---

## Files

| File | Purpose |
|------|---------|
| `train_gpt_hestia_ternary.py` | Full CUDA training script (~1600 lines) |
| `test_components.py` | Standalone component validation |
| `test_cuda_script.py` | Real-script unit tests (CPU mode) |
| `test_regressions.py` | Regression tests for GPU-discovered bugs |
| `train_mini.py` | Full pipeline smoke test (CPU) |
| `run_local_validation.py` | Local validation with real script components |
| `deploy_runpod.sh` | RunPod deployment script |

## Run Command (1×H100, STE baseline)

```bash
RUN_ID=clean_ste SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=768 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=4 EMBED_DIM=254 BITNET_GROUP_SIZE=128 ACTIVATION=leaky_relu2 \
SOFTCAP_TYPE=poly LOGIT_SOFTCAP=10 QK_GAIN_INIT=2.25 \
ROPE_TYPE=yarn YARN_MAX_LEN=2048 ROPE_BASE=5000 TIE_EMBEDDINGS=1 \
TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 \
WARMUP_STEPS=5 WARMDOWN_FRACTION=0.2 MAX_WALLCLOCK_SECONDS=540 \
ITERATIONS=10000 MATRIX_OPTIMIZER=muon MATRIX_LR=0.04 SCALAR_LR=0.02 \
TIED_EMBED_LR=0.02 MUON_BACKEND_STEPS=3 MUON_MOMENTUM=0.95 \
FP_STORAGE=FP8 HESTIA=0 STRUCTURED_2_4=0 GPTQ_TERNARY=0 EGGROLL=0 \
SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=16 TEMP_SCALING=1 \
COMPILE_MODE=default OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
python3 train_gpt_hestia_ternary.py
```

---

## Path Forward

The working configuration is standard STE ternary with ~72M params, matching the original submission. To improve on 1.1570 BPB, the most promising directions are:

1. **Fix DDP compatibility** for 8×H100 (the autograd hook assertion with torch.compile)
2. **Add XSA** (cross-sequence attention, zero parameter cost, proven -0.005+ BPB)
3. **Add BigramHash** embeddings (proven significant BPB improvement)
4. **HESTIA with Adam** instead of Muon (test if the instability is optimizer-specific)
5. **EGGROLL post-training refinement** (validated locally, needs eval-time budget integration)
6. **Block-wise GPTQ** instead of column-wise (avoid error accumulation divergence)

## Lineage

```
Ciprian-Florin Ifrim's Ternary (1.1570 BPB)
    Attempted additions (experimental):
    ├── HESTIA differentiable QAT — unstable with Muon (NaN)
    ├── 2:4 structured ternary packing — worse than base3+LZMA
    ├── GPTQ error compensation for ternary — corrupts weights
    └── EGGROLL for ternary — validated locally, not yet tested at scale
    Working additions:
    ├── FA3/SDPA auto-fallback for any GPU
    ├── LeakyReLU(0.5)² activation
    ├── Vectorized pack/unpack (100x faster)
    └── 2048 sequence training with YaRN
```
