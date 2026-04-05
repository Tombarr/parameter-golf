#!/usr/bin/env python3
"HESTIA Ternary training script for Parameter Golf. Fork of Ciprian-Florin Ifrim's ternary submission with novel additions: HESTIA QAT, 2:4 structured sparsity, GPTQ-ternary, EGGROLL."

import copy
import glob
import io
import math
import os
import random
import sys
import time
import lzma
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn_interface import flash_attn_func
    _FA3_AVAILABLE = True
except ImportError:
    _FA3_AVAILABLE = False
    def flash_attn_func(q, k, v, causal=False):
        """Fallback for non-Hopper GPUs: uses PyTorch SDPA."""
        # q: (B, S, H, D) -> (B, H, S, D)
        B, S, H, D = q.shape
        KVH = k.shape[2]
        if KVH < H:
            rep = H // KVH
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return y.transpose(1, 2)  # (B, S, H, D)

# ---------------------------------------------------------------------------
# Hyperparameters (all configurable via environment variables)
# ---------------------------------------------------------------------------
def _e(k, d, t=str):
    v = os.environ.get(k, str(d))
    if t == bool: return bool(int(v))
    return t(v)

class Hyperparameters:
    data_path = _e("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = _e("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", f"hestia_{int(time.time())}")
    seed = _e("SEED", 1337, int)
    compile_mode = _e("COMPILE_MODE", "default")
    val_batch_size = _e("VAL_BATCH_SIZE", 524288, int)
    val_loss_every = _e("VAL_LOSS_EVERY", 500, int)
    train_log_every = _e("TRAIN_LOG_EVERY", 10, int)
    iterations = _e("ITERATIONS", 10000, int)
    warmdown_fraction = _e("WARMDOWN_FRACTION", 0.2, float)
    warmup_steps = _e("WARMUP_STEPS", 20, int)
    train_batch_tokens = _e("TRAIN_BATCH_TOKENS", 524288, int)
    train_seq_len = _e("TRAIN_SEQ_LEN", 1024, int)
    max_wallclock_seconds = _e("MAX_WALLCLOCK_SECONDS", 0.0, float)
    vocab_size = _e("VOCAB_SIZE", 8192, int)
    num_layers = _e("NUM_LAYERS", 10, int)
    num_kv_heads = _e("NUM_KV_HEADS", 4, int)
    model_dim = _e("MODEL_DIM", 768, int)
    num_heads = _e("NUM_HEADS", 8, int)
    mlp_mult = _e("MLP_MULT", 4, int)
    tie_embeddings = _e("TIE_EMBEDDINGS", 1, int)
    rope_base = _e("ROPE_BASE", 5000.0, float)
    rope_type = _e("ROPE_TYPE", "yarn")
    yarn_max_len = _e("YARN_MAX_LEN", 2048, int)
    logit_softcap = _e("LOGIT_SOFTCAP", 10.0, float)
    softcap_type = _e("SOFTCAP_TYPE", "poly")
    tied_embed_init_std = _e("TIED_EMBED_INIT_STD", 0.005, float)
    qk_gain_init = _e("QK_GAIN_INIT", 2.25, float)
    activation_type = _e("ACTIVATION", "relu2")
    embed_dim = _e("EMBED_DIM", 254, int)
    bigram_hash = _e("BIGRAM_HASH", 0, bool)
    mtp_heads_count = _e("MTP_HEADS", 0, int)
    training_depth_recurrence = _e("TRAINING_DEPTH_RECURRENCE", 1, int)
    eval_depth_recurrence = _e("EVAL_DEPTH_RECURRENCE", 1, int)
    embed_lr = _e("EMBED_LR", 0.6, float)
    head_lr = _e("HEAD_LR", 0.008, float)
    adam_lr = _e("ADAM_LR", 1e-3, float)
    adam_wd = _e("ADAM_WD", 0.05, float)
    untie_at_fraction = _e("UNTIE_AT_FRACTION", 0.0, float)
    tied_embed_lr = _e("TIED_EMBED_LR", 0.05, float)
    corr_weight_lr = _e("CORR_WEIGHT_LR", 0.05, float)
    smear = _e("SMEAR", 0, bool)
    seq_len_start = _e("SEQ_LEN_START", 0, int)
    seq_schedule_fraction = _e("SEQ_SCHEDULE_FRACTION", 0.33, float)
    batch_tokens_start = _e("BATCH_TOKENS_START", 0, int)
    batch_schedule_fraction = _e("BATCH_SCHEDULE_FRACTION", 0.33, float)
    churn_log_every = _e("CHURN_LOG_EVERY", 500, int)
    matrix_lr = _e("MATRIX_LR", 0.04, float)
    scalar_lr = _e("SCALAR_LR", 0.02, float)
    muon_momentum = _e("MUON_MOMENTUM", 0.95, float)
    muon_backend_steps = _e("MUON_BACKEND_STEPS", 3, int)
    muon_wd = _e("MUON_WD", 0.0, float)
    matrix_optimizer = _e("MATRIX_OPTIMIZER", "muon")
    muon_momentum_warmup_start = _e("MUON_MOMENTUM_WARMUP_START", 0.85, float)
    muon_momentum_warmup_steps = _e("MUON_MOMENTUM_WARMUP_STEPS", 500, int)
    beta1 = _e("BETA1", 0.9, float)
    beta2 = _e("BETA2", 0.95, float)
    adam_eps = _e("ADAM_EPS", 1e-8, float)
    grad_clip_norm = _e("GRAD_CLIP_NORM", 0.0, float)
    bitnet_group_size = _e("BITNET_GROUP_SIZE", 128, int)
    sliding_eval = _e("SLIDING_EVAL", 0, bool)
    sliding_eval_stride = _e("SLIDING_EVAL_STRIDE", 16, int)
    sliding_batch_size = _e("SLIDING_BATCH_SIZE", 256, int)
    temp_scaling = _e("TEMP_SCALING", 0, bool)
    _fp_raw = os.environ.get("FP_STORAGE", "0")
    fp_storage = True if _fp_raw == "FP8" else ("fp4" if _fp_raw == "FP4" else False)
    # HESTIA parameters
    hestia_enabled = _e("HESTIA", 1, bool)
    hestia_tau_init = _e("HESTIA_TAU_INIT", 0.3, float)
    hestia_pressure_ramp = _e("HESTIA_PRESSURE_RAMP", 0.2, float)
    hestia_alpha = _e("HESTIA_ALPHA", 0.4, float)
    # 2:4 structured sparsity
    structured_2_4 = _e("STRUCTURED_2_4", 1, bool)
    sparsity_reg_weight = _e("SPARSITY_REG", 0.001, float)
    # GPTQ-ternary post-training
    gptq_ternary = _e("GPTQ_TERNARY", 1, bool)
    gptq_calib_seqs = _e("GPTQ_CALIB_SEQS", 64, int)
    gptq_calib_len = _e("GPTQ_CALIB_LEN", 2048, int)
    gptq_calib_temp = _e("GPTQ_CALIB_TEMP", 0.8, float)
    # EGGROLL eval-time
    eggroll_enabled = _e("EGGROLL", 1, bool)
    eggroll_seconds = _e("EGGROLL_SECONDS", 60, int)
    eggroll_candidates = _e("EGGROLL_CANDIDATES", 1024, int)

CTP = ("attn_scale","attn_scales","mlp_scale","mlp_scales","resid_mix","resid_mixes",
       "q_gain","diff_lambda","skip_weight","skip_weights","vocab_bias","refiner.gate")

# ---------------------------------------------------------------------------
# 2:4 Structured Ternary Packing
# ---------------------------------------------------------------------------
PATTERNS_24 = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERNS_24)}

def enforce_2_4_structure(w_continuous: Tensor) -> Tensor:
    """Enforce 2:4 sparsity on continuous weights → ternary {-1,0,1} with exactly 2 nonzeros per 4."""
    orig_shape = w_continuous.shape
    flat = w_continuous.reshape(-1)
    # Pad to multiple of 4
    pad4 = (4 - flat.numel() % 4) % 4
    if pad4 > 0:
        flat = F.pad(flat, (0, pad4))
    groups = flat.reshape(-1, 4)
    _, top2 = groups.abs().topk(2, dim=-1)
    result = torch.zeros_like(groups)
    signs = groups.sign()
    signs[signs == 0] = 1.0
    result.scatter_(1, top2, signs.gather(1, top2))
    return result.reshape(-1)[:w_continuous.numel()].reshape(orig_shape)

def pack_2_4_ternary(q: Tensor) -> tuple[bytes, int]:
    """Pack 2:4 structured ternary weights. Each group of 4 → 1 byte."""
    flat = q.reshape(-1)
    # Pad to multiple of 4
    pad4 = (4 - flat.numel() % 4) % 4
    if pad4 > 0:
        flat = F.pad(flat, (0, pad4))
    groups = flat.reshape(-1, 4).to(torch.int8).cpu().numpy()
    n_groups = groups.shape[0]
    packed = np.zeros(n_groups, dtype=np.uint8)
    for i in range(n_groups):
        g = groups[i]
        nz = tuple(np.where(g != 0)[0])
        if len(nz) != 2:
            nz = (0, 1)
        pidx = PATTERN_TO_IDX.get(nz, 0)
        s0 = 1 if g[nz[0]] > 0 else 0
        s1 = 1 if g[nz[1]] > 0 else 0
        packed[i] = pidx * 4 + s0 * 2 + s1
    return packed.tobytes(), n_groups

def unpack_2_4_ternary(data: bytes, n_groups: int) -> Tensor:
    """Unpack 2:4 structured ternary weights from packed bytes."""
    packed = np.frombuffer(data, dtype=np.uint8)[:n_groups]
    result = np.zeros((n_groups, 4), dtype=np.int8)
    for i in range(n_groups):
        val = int(packed[i])
        pidx, signs = val // 4, val % 4
        s0 = 1 if (signs & 2) else -1
        s1 = 1 if (signs & 1) else -1
        p0, p1 = PATTERNS_24[pidx]
        result[i, p0] = s0
        result[i, p1] = s1
    return torch.from_numpy(result.reshape(-1))

# ---------------------------------------------------------------------------
# Standard ternary packing (base-3, fallback if 2:4 disabled)
# ---------------------------------------------------------------------------
def pack_ternary(q: Tensor):
    f = (q.reshape(-1).to(torch.int8) + 1).cpu().numpy()
    n = len(f)
    p = (5 - n % 5) % 5
    if p: f = np.concatenate([f, np.zeros(p, dtype=np.int8)])
    g = f.reshape(-1, 5).astype(np.uint8)
    return (g[:,0] + g[:,1]*3 + g[:,2]*9 + g[:,3]*27 + g[:,4]*81).tobytes(), n

def unpack_ternary(data: bytes, n: int) -> Tensor:
    v = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    t = np.zeros((len(v), 5), dtype=np.int8)
    for i in range(5): t[:,i] = v % 3; v //= 3
    return torch.from_numpy(t.reshape(-1)[:n].astype(np.int8) - 1)

# ---------------------------------------------------------------------------
# FP8/FP4 quantization (unchanged from base)
# ---------------------------------------------------------------------------
def quantize_to_int4(t: Tensor) -> tuple[Tensor, Tensor, list]:
    t32 = t.float()
    orig_shape = t32.shape
    if t32.ndim < 2: t32 = t32.unsqueeze(0)
    absmax = t32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 7.0
    q = torch.clamp(torch.round(t32 / scale), -7, 7).to(torch.int8)
    flat = q.reshape(-1)
    if flat.numel() % 2 != 0: flat = F.pad(flat, (0, 1))
    low = (flat[0::2] + 8).to(torch.uint8)
    high = (flat[1::2] + 8).to(torch.uint8)
    return low | (high << 4), scale.half().squeeze(-1), list(orig_shape)

def dequantize_from_int4(packed: Tensor, scale: Tensor, shape: list) -> Tensor:
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    flat = torch.zeros(packed.numel() * 2, dtype=torch.int8)
    flat[0::2] = low; flat[1::2] = high
    numel = 1
    for s in shape: numel *= s
    flat = flat[:numel].float()
    if len(shape) <= 1: return (flat * scale.float().squeeze()).reshape(shape)
    return (flat.reshape(-1, shape[-1]) * scale.float().unsqueeze(-1)).reshape(shape)

# ---------------------------------------------------------------------------
# HESTIA: Differentiable Softmax Relaxation for Ternary QAT
# ---------------------------------------------------------------------------
# These globals are ONLY used for non-compiled code paths and initialization.
# The compiled path reads from tensor buffers on HestiaTernaryLinear.
_HESTIA_TAU = 0.3
_HESTIA_PRESSURE = 0.0
_HESTIA_ENABLED = True

def hestia_soft_quantize(w_grouped: Tensor, scale: Tensor, tau: Tensor) -> Tensor:
    """Softmax relaxation over ternary grid {-1, 0, 1}.
    tau must be a scalar Tensor (not Python float) for torch.compile compatibility."""
    grid = torch.tensor([-1.0, 0.0, 1.0], device=w_grouped.device, dtype=w_grouped.dtype)
    w_norm = (w_grouped / scale).unsqueeze(-1)
    logits = -((w_norm - grid) ** 2) / tau.clamp(min=1e-7)
    probs = F.softmax(logits, dim=-1)
    return scale * (probs * grid).sum(dim=-1)

def hestia_ternary_forward(w: Tensor, group_size: int, tau: Tensor, pressure: Tensor,
                           sensitivity: float = 0.0) -> Tensor:
    """HESTIA-aware ternary forward pass. Replaces standard STE.
    tau and pressure are scalar Tensors (buffers) for torch.compile compatibility.
    Uses torch.where instead of Python if/else to avoid graph breaks."""
    w_bf = w.bfloat16()
    g = group_size
    w_g = w_bf.reshape(-1, g)
    scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)

    # Always compute STE path (cheap)
    q = (w_g / scale).round().clamp(-1, 1)
    w_ste = w_g + ((q * scale) - w_g).detach()

    # Always compute HESTIA soft path (only adds cost when pressure > 0)
    tau_eff = tau * math.exp(0.4 * sensitivity)
    w_soft = hestia_soft_quantize(w_g, scale, tau_eff.clamp(min=1e-7))
    w_hestia = (1.0 - pressure) * w_g + pressure * w_soft

    # Blend: use HESTIA when pressure > 0, otherwise STE
    # torch.where avoids Python branching → no graph break
    w_eff = torch.where(pressure > 0, w_hestia, w_ste)

    return w_eff.reshape(w.shape)

# ---------------------------------------------------------------------------
# GPTQ-Ternary: Post-training error compensation on ternary grid
# ---------------------------------------------------------------------------
def gptq_ternary_quantize(W: Tensor, H: Tensor, scale: Tensor) -> Tensor:
    """
    GPTQ error compensation adapted for ternary {-1,0,1}.
    W: (out, in) float, H: (in, in) Hessian, scale: (out, 1) per-row.
    Returns: (out, in) ternary values {-1,0,1}.
    """
    W = W.clone().float()
    n_out, n_in = W.shape
    Q = torch.zeros_like(W)
    damp = 0.01 * H.diag().mean().clamp(min=1e-6)
    H = H + damp * torch.eye(n_in, device=H.device, dtype=H.dtype)
    scale_sq = scale.squeeze(-1)
    for j in range(n_in):
        w_col = W[:, j]
        q_val = (w_col / scale_sq).round().clamp(-1, 1)
        q_col = q_val * scale_sq
        Q[:, j] = q_val
        error = w_col - q_col
        if j + 1 < n_in:
            h_jj = H[j, j].clamp(min=1e-8)
            W[:, j+1:] += error.unsqueeze(1) * (H[j, j+1:] / h_jj).unsqueeze(0)
    return Q

def collect_activations_for_gptq(model, calib_tokens, device):
    """Run calibration tokens through model, collect per-layer input activations."""
    activations = {}
    hooks = []
    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if name not in activations:
                activations[name] = []
            activations[name].append(x.reshape(-1, x.shape[-1]))
        return hook_fn
    # Register hooks on all TernaryLinear layers
    for name, module in model.named_modules():
        if isinstance(module, (HestiaTernaryLinear,)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    # Forward pass
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, calib_tokens.shape[0], 8):
            batch = calib_tokens[i:i+8].to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            model(x, y)
    for h in hooks:
        h.remove()
    # Compute Hessians
    hessians = {}
    for name, acts in activations.items():
        X = torch.cat(acts, dim=0)
        n = X.shape[0]
        # Subsample if too large
        if n > 4096:
            idx = torch.randperm(n)[:4096]
            X = X[idx]
        hessians[name] = (X.T @ X) / X.shape[0]
    return hessians

# ---------------------------------------------------------------------------
# EGGROLL: Gradient-free ternary refinement
# ---------------------------------------------------------------------------
def eggroll_refine(model, val_tokens, device, max_seconds=60, candidates_per_iter=1024):
    """
    EGGROLL adapted for ternary: gradient-free search over {-1,0,1} to minimize val loss.
    Runs within the eval time budget.
    """
    model.eval()
    total_flips = 0
    t0 = time.time()

    # Get a small eval batch for fast loss computation
    seq_len = 1024
    n_seqs = min(32, (val_tokens.numel() - 1) // seq_len)
    eval_x = val_tokens[:n_seqs * seq_len].reshape(n_seqs, seq_len).to(device)
    eval_y = val_tokens[1:n_seqs * seq_len + 1].reshape(n_seqs, seq_len).to(device)

    # Compute baseline loss
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        base_loss = model(eval_x, eval_y).item()

    # Iterate over ternary parameters
    ternary_params = [(n, p) for n, p in model.named_parameters()
                      if p.ndim == 2 and p.numel() > 65536
                      and "tok_emb" not in n and "embed" not in n and "bigram" not in n]

    current_loss = base_loss
    iteration = 0

    while time.time() - t0 < max_seconds and ternary_params:
        # Pick a random parameter
        name, param = random.choice(ternary_params)
        with torch.no_grad():
            g = 128  # group size
            w = param.data.float()
            w_g = w.reshape(-1, g)
            scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
            q = (w_g / scale).round().clamp(-1, 1)

            # Try random flips
            n_total = q.numel()
            indices = torch.randint(0, n_total, (candidates_per_iter,), device=q.device)
            rows = indices // g
            cols = indices % g

            for k in range(candidates_per_iter):
                r, c = rows[k].item(), cols[k].item()
                old_val = q[r, c].item()
                # Try alternatives
                for new_val in [-1.0, 0.0, 1.0]:
                    if new_val == old_val:
                        continue
                    q[r, c] = new_val
                    # Reconstruct weight and evaluate
                    param.data.copy_((q * scale).reshape(w.shape).bfloat16())
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        new_loss = model(eval_x, eval_y).item()
                    if new_loss < current_loss:
                        current_loss = new_loss
                        old_val = new_val
                        total_flips += 1
                    else:
                        q[r, c] = old_val
                        param.data.copy_((q * scale).reshape(w.shape).bfloat16())

        iteration += 1
        if iteration % 5 == 0:
            elapsed = time.time() - t0
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print(f"  eggroll iter={iteration} flips={total_flips} loss={current_loss:.6f} "
                      f"dt={elapsed:.1f}s", flush=True)

    return total_flips, current_loss

# ---------------------------------------------------------------------------
# Hutch++ Hessian trace estimation (for HESTIA scheduling)
# ---------------------------------------------------------------------------
def estimate_sensitivities(model, calib_batch, device, group_size=128):
    """Compute per-tensor Hessian trace for HESTIA temperature scheduling."""
    sensitivities = {}
    hooks = []
    acts = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            acts[name] = x.reshape(-1, x.shape[-1])
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, HestiaTernaryLinear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = calib_batch[:, :-1]
        y = calib_batch[:, 1:]
        model(x, y)

    for h in hooks:
        h.remove()

    # Estimate trace via Hutchinson
    for name, X in acts.items():
        n = X.shape[1]
        H = (X.T @ X) / X.shape[0]
        traces = []
        for _ in range(10):
            v = torch.sign(torch.randn(n, device=device, dtype=torch.float32))
            traces.append((v * (H @ v)).sum().item())
        sensitivities[name] = sum(traces) / len(traces)

    # Normalize to z-scores
    if sensitivities:
        vals = list(sensitivities.values())
        mean_s = sum(vals) / len(vals)
        std_s = max((sum((v - mean_s)**2 for v in vals) / len(vals)) ** 0.5, 1e-8)
        sensitivities = {k: (v - mean_s) / std_s for k, v in sensitivities.items()}

    model.train()
    return sensitivities

# ---------------------------------------------------------------------------
# State dict serialization (2:4 ternary + fp8/fp16)
# ---------------------------------------------------------------------------
def q_sd(state_dict: dict, group_size: int = 128, fp_storage=False,
         use_2_4: bool = True, gptq_hessians: dict = None) -> tuple[dict, dict]:
    """Quantize state dict. Ternary for large 2D weights, fp8/fp16 for rest.
    Optionally uses GPTQ error compensation and 2:4 structured packing."""
    quantized = {}
    stats = {"ternary_params": 0, "ternary_bytes": 0, "fp_params": 0, "fp_bytes": 0}

    for name, tensor in state_dict.items():
        if "mtp_heads" in name:
            continue
        t = tensor.detach().cpu().float().contiguous()
        t_orig_shape = list(t.shape)

        if t.ndim == 3:
            t = t.reshape(t.shape[0], -1)

        is_ternary = (
            t.ndim == 2 and t.numel() > 65_536
            and "tok_emb" not in name and "lm_head" not in name
            and "embed_proj" not in name and "bigram_emb" not in name
            and "lm_head_correction" not in name and "lm_head_U" not in name
            and "lm_head_V" not in name
        )

        if is_ternary:
            pad = (group_size - t.shape[1] % group_size) % group_size
            t_padded = F.pad(t, (0, pad)) if pad > 0 else t
            t_grouped = t_padded.reshape(-1, group_size)
            scale = t_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()

            if gptq_hessians and name in gptq_hessians:
                # GPTQ error compensation before quantization
                H = gptq_hessians[name].cpu().float()
                row_scale = scale.reshape(t.shape[0], -1).mean(dim=1, keepdim=True)
                q_raw = gptq_ternary_quantize(t, H, row_scale)
                q = F.pad(q_raw, (0, pad)) if pad > 0 else q_raw
                q = q.reshape(-1, group_size)
            else:
                q = (t_grouped / scale).round().clamp(-1, 1).to(torch.int8)

            if use_2_4:
                # Enforce 2:4 structure and pack
                q_flat = q.reshape(-1).float()
                q_24 = enforce_2_4_structure(q_flat).to(torch.int8)
                packed_bytes, n_groups = pack_2_4_ternary(q_24)
                quantized[name] = {
                    "type": "ternary_2_4",
                    "packed": packed_bytes,
                    "n_groups": n_groups,
                    "scale": scale.half().squeeze(-1),
                    "shape": list(t.shape),
                    "padded_cols": t_padded.shape[1],
                    "group_size": group_size,
                    "orig_shape": t_orig_shape,
                }
                stats["ternary_params"] += t.numel()
                stats["ternary_bytes"] += len(packed_bytes) + scale.numel() * 2
            else:
                # Standard base-3 packing (fallback)
                packed_bytes, n_trits = pack_ternary(q)
                quantized[name] = {
                    "type": "ternary",
                    "packed": packed_bytes,
                    "scale": scale.half().squeeze(-1),
                    "shape": list(t.shape),
                    "padded_cols": t_padded.shape[1],
                    "group_size": group_size,
                    "n_trits": n_trits,
                    "orig_shape": t_orig_shape,
                }
                stats["ternary_params"] += t.numel()
                stats["ternary_bytes"] += len(packed_bytes) + scale.numel() * 2
        elif fp_storage and t.ndim == 2:
            quantized[name] = {"type": "fp8", "data": t.to(torch.float8_e4m3fn)}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel()
        else:
            quantized[name] = {"type": "fp16", "data": t.half()}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel() * 2

    return quantized, stats

def deq_sd(quantized: dict, target_dtype=torch.bfloat16):
    """Reconstruct full-precision state dict from quantized representation."""
    out = {}
    for name, entry in quantized.items():
        if entry["type"] == "ternary_2_4":
            q = unpack_2_4_ternary(entry["packed"], entry["n_groups"])
            q = q.float()
            g = entry["group_size"]
            scale = entry["scale"].float().unsqueeze(-1)
            padded_cols = entry["padded_cols"]
            shape = entry["shape"]
            # Reshape to padded matrix form
            total_padded = shape[0] * padded_cols
            q_padded = q[:total_padded].reshape(-1, g)
            # Reconstruct: scale * ternary_value
            # But 2:4 ternary has no zero-fraction mismatch — values are exactly {-1,0,1}
            t = (q_padded * scale).reshape(shape[0], padded_cols)
            result = t[:shape[0], :shape[1]].to(target_dtype)
            orig = entry.get("orig_shape")
            out[name] = result.reshape(orig).contiguous() if orig and orig != shape else result.contiguous()
        elif entry["type"] == "ternary":
            q = unpack_ternary(entry["packed"], entry["n_trits"])
            q = q.float().reshape(-1, entry["group_size"])
            scale = entry["scale"].float().unsqueeze(-1)
            q_absmean = q.abs().mean(-1, keepdim=True).clamp(min=1e-8)
            t = (q * (scale / q_absmean)).reshape(-1, entry["padded_cols"])
            shape = entry["shape"]
            result = t[:shape[0], :shape[1]].to(target_dtype)
            orig = entry.get("orig_shape")
            out[name] = result.reshape(orig).contiguous() if orig and orig != shape else result.contiguous()
        elif entry["type"] == "fp8":
            out[name] = entry["data"].to(torch.float32).to(target_dtype).contiguous()
        elif entry["type"] == "fp4":
            out[name] = dequantize_from_int4(entry["packed"], entry["scale"], entry["shape"]).to(target_dtype).contiguous()
        else:
            out[name] = entry["data"].to(target_dtype).contiguous()
    return out

# ---------------------------------------------------------------------------
# Ternary diagnostics
# ---------------------------------------------------------------------------
def tern_stats(model: nn.Module, group_size: int = 128):
    total = zeros = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and ("weight" in name) and p.shape[0] > 1:
                w = p.detach().float().reshape(-1, group_size)
                scale = w.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
                q = (w / scale).round().clamp(-1, 1)
                zeros += int((q == 0).sum().item())
                total += int(q.numel())
    return {"zero_frac": zeros / max(total, 1), "total_weights": total}

_prev_committed: dict = {}
def churn_fn(model: nn.Module, group_size: int = 128):
    global _prev_committed
    total = flipped = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and ("weight" in name) and p.shape[0] > 1:
                w = p.detach().float().reshape(-1, group_size)
                scale = w.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
                q = (w / scale).round().clamp(-1, 1).cpu().numpy()
                if name in _prev_committed:
                    flipped += int(np.sum(q != _prev_committed[name]))
                    total += q.size
                _prev_committed[name] = q
    return flipped / max(total, 1)

# ---------------------------------------------------------------------------
# Muon optimizer (Newton-Schulz orthogonalized momentum)
# ---------------------------------------------------------------------------
def ns_orth(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, wd=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, wd=wd))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = F.rms_norm(g.float(), (g.size(-1),)).bfloat16()
                    g = ns_orth(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("wd", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0: p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def ld_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = ld_shard(self.files[0])
        self.pos = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = ld_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start:start + per_rank_span].pin_memory().to(self.device, non_blocking=True).to(torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x, y

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

def apply_qat_ste(w, fp_storage):
    if not fp_storage: return w
    if fp_storage == "fp4":
        absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 7.0
        q = torch.clamp(torch.round(w / scale), -7.0, 7.0)
        w_sim = q * scale
        return (w_sim - w).detach() + w
    elif fp_storage is True or fp_storage == "fp8":
        w_sim = w.to(torch.float8_e4m3fn).to(w.dtype)
        return (w_sim - w).detach() + w
    return w

class QATLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=False, fp_storage=False):
        super().__init__(in_f, out_f, bias=bias)
        self.fp_storage = fp_storage
    def forward(self, x):
        w = apply_qat_ste(self.weight, self.fp_storage)
        return F.linear(x, w.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class QATEmbedding(nn.Embedding):
    def __init__(self, num_emb, emb_dim, fp_storage=False):
        super().__init__(num_emb, emb_dim)
        self.fp_storage = fp_storage
    def forward(self, input):
        w = apply_qat_ste(self.weight, self.fp_storage)
        return F.embedding(input, w, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

class HestiaTernaryLinear(nn.Linear):
    """Ternary linear with HESTIA differentiable QAT (replaces TernaryLinear).
    Uses tensor buffers for tau/pressure so torch.compile doesn't recompile on value changes."""
    def __init__(self, in_features, out_features, bias=False, group_size=128):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = group_size
        self.sensitivity = 0.0  # set by Hutch++ estimation
        # Tensor buffers — updated by the training loop, read by torch.compile
        self.register_buffer("hestia_tau", torch.tensor(0.3), persistent=False)
        self.register_buffer("hestia_pressure", torch.tensor(0.0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        w_ternary = hestia_ternary_forward(
            self.weight, self.group_size,
            self.hestia_tau, self.hestia_pressure,
            self.sensitivity)
        return F.linear(x, w_ternary.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)

class NormedHestiaTernaryLinear(HestiaTernaryLinear):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.rms_norm(x, (x.size(-1),)))

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CTP)) and param.dtype != torch.float32:
                param.data = param.data.float()

# ---------------------------------------------------------------------------
# Rotary embeddings (YaRN)
# ---------------------------------------------------------------------------
class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, no_cache=False, rope_type="rope",
                 yarn_max_len=4096, train_seq_len=1024):
        super().__init__()
        self.no_cache = no_cache
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if rope_type == "yarn":
            scale = train_seq_len / yarn_max_len
            freq_idx = torch.arange(0, dim, 2, dtype=torch.float32)
            ramp = torch.clamp((freq_idx / dim - 0.25) / 0.75, 0.0, 1.0)
            inv_freq = inv_freq / (ramp * (1.0 / scale - 1.0) + 1.0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self.no_cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            return freqs.cos()[None,:,None,:].to(dtype), freqs.sin()[None,:,None,:].to(dtype)
        if (self._cos_cached is None or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None,:,None,:]
            self._sin_cached = freqs.sin()[None,:,None,:]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype), self._sin_cached.to(dtype)

def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 group_size=128, no_cache=False, rope_type="rope",
                 yarn_max_len=4096, train_seq_len=1024):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.c_qkv = HestiaTernaryLinear(dim, self.q_size + 2 * self.kv_size, group_size=group_size)
        self.proj = NormedHestiaTernaryLinear(dim, dim, group_size=group_size)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, no_cache=no_cache,
                             rope_type=rope_type, yarn_max_len=yarn_max_len,
                             train_seq_len=train_seq_len)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        qkv = self.c_qkv(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, group_size=128, activation="relu2"):
        super().__init__()
        hidden = mlp_mult * dim
        self.activation = activation
        self.fc = HestiaTernaryLinear(dim, hidden, group_size=group_size)
        self.proj = NormedHestiaTernaryLinear(hidden, dim, group_size=group_size)
        self.proj._zero_init = True

    def forward(self, x):
        if self.activation == "relu2":
            return self.proj(torch.relu(self.fc(x)).square())
        elif self.activation == "leaky_relu2":
            return self.proj(F.leaky_relu(self.fc(x), 0.5).square())
        else:
            return self.proj(torch.relu(self.fc(x)))

class SmearModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        cumsum = x.cumsum(dim=1)
        counts = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        smeared = cumsum / counts
        gate = torch.tanh(self.gate.to(dtype=x.dtype))
        return x + gate * (smeared - x)

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 group_size=128, activation="relu2", no_cache=False, smear=False,
                 rope_type="rope", yarn_max_len=4096, train_seq_len=1024):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base,
                                        qk_gain_init, group_size, no_cache,
                                        rope_type, yarn_max_len, train_seq_len)
        self.mlp = MLP(dim, mlp_mult, group_size, activation)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.smear = SmearModule(dim) if smear else None

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0] * x + mix[1] * x0
        n = self.attn_norm(x)
        x = x + self.attn_scale.to(dtype=x.dtype) * self.attn(n)
        x = x + self.mlp_scale.to(dtype=x.dtype) * self.mlp(self.mlp_norm(x))
        if self.smear is not None: x = self.smear(x)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init,
                 group_size=128, activation="relu2", mtp_heads_count=0, embed_dim=0,
                 training_depth_recurrence=1, fp_storage=False, bigram_hash=False,
                 softcap_type="poly", no_cache=False, smear=False, rope_type="rope",
                 yarn_max_len=4096, train_seq_len=1024):
        super().__init__()
        self.training_depth_recurrence = training_depth_recurrence
        self.fp_storage = fp_storage
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.softcap_type = softcap_type
        self.embed_dim = embed_dim if embed_dim > 0 else model_dim
        self.tok_emb = QATEmbedding(vocab_size, self.embed_dim, fp_storage=fp_storage)
        self.bigram_emb = QATEmbedding(vocab_size, self.embed_dim, fp_storage=fp_storage) if bigram_hash else None
        if self.bigram_emb is not None: nn.init.zeros_(self.bigram_emb.weight)
        self.lm_head_correction = nn.Parameter(
            torch.zeros(vocab_size, self.embed_dim)) if tie_embeddings == 2 else None
        self.embed_proj = QATLinear(self.embed_dim, model_dim, bias=False, fp_storage=fp_storage) if self.embed_dim != model_dim else None
        self.embed_proj_rev = QATLinear(model_dim, self.embed_dim, bias=False, fp_storage=fp_storage) if self.embed_dim != model_dim else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  group_size, activation, no_cache, smear, rope_type, yarn_max_len, train_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.mtp_heads = nn.ModuleList([
            nn.Linear(model_dim, vocab_size, bias=False) for _ in range(mtp_heads_count)
        ])
        for h in self.mtp_heads: nn.init.zeros_(h.weight)
        self.lm_head = QATLinear(model_dim, vocab_size, bias=False, fp_storage=fp_storage)
        self.lm_head._zero_init = True
        if self.lm_head is not None and tie_embeddings:
            self.lm_head.weight.requires_grad_(False)
        self.vocab_bias = nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))
        self._init_weights(tied_embed_init_std)

    def set_hestia_params(self, tau: float, pressure: float):
        """Update HESTIA buffers on all ternary layers. Call before each training step."""
        for m in self.modules():
            if isinstance(m, HestiaTernaryLinear):
                m.hestia_tau.fill_(tau)
                m.hestia_pressure.fill_(pressure)

    def _init_weights(self, tied_embed_init_std):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, HestiaTernaryLinear) and not getattr(module, "_zero_init", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _compute_logits(self, x):
        if self.tie_embeddings:
            proj = self.embed_proj_rev(x) if self.embed_proj_rev is not None else x
            weight = self.tok_emb.weight
            if self.lm_head_correction is not None:
                weight = weight + self.lm_head_correction
            logits_raw = F.linear(proj, weight.to(x.dtype))
        else:
            logits_raw = self.lm_head(x)
        return logits_raw + self.vocab_bias.to(x.dtype)

    def _softcap(self, logits):
        s = self.logit_softcap
        if self.softcap_type == "tanh":
            return s * torch.tanh(logits / s)
        x_sc = torch.clamp(logits / s, -2.0, 2.0)
        x2 = x_sc * x_sc
        return s * torch.clamp(x_sc * (1.0 - x2 / 3.0 + x2 * x2 / 15.0), -1.0, 1.0)

    def forward(self, input_ids, target_ids, reduction="mean", temperature=1.0):
        x = self.tok_emb(input_ids).float()
        if self.bigram_emb is not None:
            prev = F.pad(input_ids[:, :-1], (1, 0), value=0)
            x = x + self.bigram_emb(prev).float()
        if self.embed_proj is not None: x = self.embed_proj(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        # U-Net encoder/decoder with skip connections
        skips = []
        for i in range(self.num_encoder_layers):
            for _ in range(max(1, self.training_depth_recurrence)):
                x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype) * skips.pop()
            for _ in range(max(1, self.training_depth_recurrence)):
                x = self.blocks[bi](x, x0)
        x_normed = self.final_norm(x)
        x_flat = x_normed.reshape(-1, x_normed.size(-1))
        targets = target_ids.reshape(-1)
        logits = self._softcap(self._compute_logits(x_flat))
        if temperature != 1.0:
            logits = logits / temperature
        if reduction == "none":
            return F.cross_entropy(logits.float(), targets, reduction="none").reshape(input_ids.shape)
        logits_f = logits.float()
        lse = torch.logsumexp(logits_f, dim=-1)
        target_logits = logits_f.gather(1, targets.unsqueeze(1)).squeeze(1)
        main_loss = (lse - target_logits).mean() + 1e-4 * (lse ** 2).mean()
        if self.training and len(self.mtp_heads) > 0:
            mtp_loss = torch.zeros((), device=main_loss.device)
            for k, head in enumerate(self.mtp_heads):
                shift = k + 2
                if target_ids.shape[1] > shift:
                    mtp_tgt = target_ids[:, shift:].reshape(-1)
                    mtp_in = x_normed[:, :target_ids.shape[1] - shift, :].reshape(-1, x_normed.shape[-1])
                    mtp_loss = mtp_loss + F.cross_entropy(head(mtp_in).float(), mtp_tgt, reduction="mean")
            main_loss = main_loss + 0.1 * mtp_loss / len(self.mtp_heads)
        return main_loss

    @torch.no_grad()
    def generate(self, n_seqs, max_len, temperature=0.8, device=None):
        """Autoregressive generation for GPTQ calibration."""
        if device is None: device = next(self.parameters()).device
        self.eval()
        tokens = torch.zeros(n_seqs, 1, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            x = self.tok_emb(tokens).float()
            if self.embed_proj is not None: x = self.embed_proj(x)
            x = F.rms_norm(x, (x.size(-1),))
            x0 = x
            skips = []
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0); skips.append(x)
            for i in range(self.num_decoder_layers):
                bi = self.num_encoder_layers + i
                if skips: x = x + self.skip_weights[i].to(dtype=x.dtype) * skips.pop()
                x = self.blocks[bi](x, x0)
            x = self.final_norm(x)
            logits = self._softcap(self._compute_logits(x[:, -1:, :].reshape(-1, x.size(-1))))
            probs = F.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_tok], dim=1)
        self.train()
        return tokens

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def build_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id): base_bytes_np[token_id] = 1; continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"): has_leading_space_np[token_id] = True; piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def ld_val(pattern, seq_len, max_tok=int(os.environ.get("VAL_MAX_TOKENS", 500000))):
    files = sorted(glob.glob(pattern))
    assert files, f"No files: {pattern}"
    tok = torch.cat([ld_shard(Path(p)) for p in files]).contiguous()
    if max_tok > 0: tok = tok[:max_tok + 1]
    u = ((tok.numel() - 1) // seq_len) * seq_len
    return tok[:u + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, temperature=1.0):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = max(1, local_batch_tokens // args.train_seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(seq_start, seq_end, local_batch_seqs):
            be = min(bs + local_batch_seqs, seq_end)
            rs = bs * args.train_seq_len; re = be * args.train_seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bloss = model(x, y, temperature=temperature).detach()
            n = float(y.numel())
            loss_sum += bloss.to(torch.float64) * n; token_count += n
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = loss_sum / token_count
    bpb = (val_loss.item() / math.log(2.0)) * (token_count.item() / byte_count.item())
    model.train()
    return float(val_loss.item()), float(bpb)

def eval_val_sliding(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride=64, temperature=1.0):
    seq_len = args.train_seq_len
    batch_size = args.sliding_batch_size
    total_tokens = val_tokens.numel() - 1
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    all_starts = list(range(0, total_tokens - seq_len, stride))
    my_starts = all_starts[rank::world_size]
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(my_starts), batch_size):
            batch_starts = my_starts[i:i + batch_size]
            starts_t = torch.tensor(batch_starts, dtype=torch.int64)
            offsets = torch.arange(seq_len + 1, dtype=torch.int64)
            indices = starts_t.unsqueeze(1) + offsets.unsqueeze(0)
            local_batch = val_tokens[indices].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local_batch[:, :-1], local_batch[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ptl = model(x, y, reduction="none", temperature=temperature).detach()
            for b, start in enumerate(batch_starts):
                sf = 0 if start == 0 else seq_len - stride
                scored = ptl[b, sf:]
                sx, sy = x[b, sf:], y[b, sf:]
                loss_sum += scored.to(torch.float64).sum()
                token_count += scored.numel()
                tb = base_bytes_lut[sy].to(torch.int16)
                tb += (has_leading_space_lut[sy] & ~is_boundary_token_lut[sx]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = loss_sum / token_count
    bpb = (val_loss.item() / math.log(2.0)) * (token_count.item() / byte_count.item())
    model.train()
    return float(val_loss.item()), float(bpb)

def find_temp(args, model, rank, world_size, device, grad_accum_steps,
              calib_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    best_t, best_loss = 1.0, float("inf")
    for t in [0.85, 0.90, 0.95, 1.00, 1.05]:
        loss, _ = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                           calib_tokens, base_bytes_lut, has_leading_space_lut,
                           is_boundary_token_lut, temperature=t)
        if loss < best_loss: best_loss = loss; best_t = t
    return best_t

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main() -> None:
    global _HESTIA_TAU, _HESTIA_PRESSURE, _HESTIA_ENABLED
    args = Hyperparameters()
    code = Path(__file__).read_text(encoding="utf-8")
    _HESTIA_ENABLED = args.hestia_enabled

    if args.matrix_optimizer != "adamw":
        global ns_orth
        ns_orth = torch.compile(ns_orth)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs("logs/cuda/", exist_ok=True)
    logfile = f"logs/cuda/{args.run_id}.txt" if master_process else None
    if master_process: print(logfile)
    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = ld_val(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(sp, args.vocab_size, device)

    # --- Model ---
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        group_size=args.bitnet_group_size, activation=args.activation_type,
        mtp_heads_count=args.mtp_heads_count, embed_dim=args.embed_dim,
        training_depth_recurrence=args.training_depth_recurrence, fp_storage=args.fp_storage,
        bigram_hash=args.bigram_hash, softcap_type=args.softcap_type,
        no_cache=(args.compile_mode == "reduce-overhead"),
        smear=args.smear, rope_type=args.rope_type, yarn_max_len=args.yarn_max_len,
        train_seq_len=args.train_seq_len,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, nn.Linear): module.float()
    restore_low_dim_params_to_fp32(base_model)
    if base_model.lm_head is not None and args.tie_embeddings:
        base_model.lm_head.weight.requires_grad_(False)

    # --- Hutch++ Sensitivity Estimation ---
    if args.hestia_enabled and master_process:
        log0("Computing Hutch++ sensitivities...")
        train_loader_tmp = DistributedTokenLoader(args.train_files, rank, world_size, device)
        n_calib_seqs = 8
        calib_tokens_needed = n_calib_seqs * (args.train_seq_len + 1)
        calib_batch = train_loader_tmp.stream.take(calib_tokens_needed)
        calib_batch = calib_batch[:calib_tokens_needed].to(device).long().reshape(n_calib_seqs, -1)
        sensitivities = estimate_sensitivities(base_model, calib_batch, device, args.bitnet_group_size)
        for name, module in base_model.named_modules():
            if isinstance(module, HestiaTernaryLinear) and name in sensitivities:
                module.sensitivity = sensitivities[name]
                log0(f"  {name}: sensitivity={sensitivities[name]:.3f}", console=False)
        del train_loader_tmp, calib_batch
        torch.cuda.empty_cache()

    torch._dynamo.config.optimize_ddp = False
    compiled_model = torch.compile(base_model, mode=args.compile_mode if args.compile_mode != "default" else None)
    use_find_unused = args.untie_at_fraction > 0 or args.mtp_heads_count > 0 or not args.tie_embeddings
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False,
                find_unused_parameters=use_find_unused, static_graph=not use_find_unused,
                gradient_as_bucket_view=True) if distributed else compiled_model

    # --- Optimizers ---
    _excl = {"tok_emb.weight", "lm_head.weight", "lm_head_correction"}
    all_other = [(n, p) for n, p in base_model.named_parameters() if not any(e in n for e in _excl)]
    matrix_params = [p for n, p in all_other if p.ndim == 2 and not any(pat in n for pat in CTP)]
    scalar_params = [p for n, p in all_other if p.ndim < 2 or any(pat in n for pat in CTP)]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_tok = torch.optim.Adam([{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
                               betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps, wd=args.muon_wd)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                  betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_head = torch.optim.Adam([{"params": [base_model.lm_head.weight], "lr": 0.0, "base_lr": 0.0}],
                                betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar, opt_head]
    if base_model.lm_head_correction is not None:
        opt_corr = torch.optim.Adam([{"params": [base_model.lm_head_correction],
                                      "lr": args.corr_weight_lr, "base_lr": args.corr_weight_lr}],
                                    betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.append(opt_corr)

    log0("--- Hyperparameters ---", console=False)
    log0(" ".join(f"{a}={getattr(args,a)}" for a in sorted(dir(args))
                  if not a.startswith("_") and a not in ("train_files","val_files") and not callable(getattr(args,a))),
         console=False)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"params:{n_params} L:{args.num_layers} d:{args.model_dim} h:{args.num_heads} "
         f"kv:{args.num_kv_heads} ws:{world_size} ga:{grad_accum_steps} s:{args.seed} "
         f"hestia:{args.hestia_enabled} 2:4:{args.structured_2_4} gptq:{args.gptq_ternary}")

    # --- Data loader ---
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_fraction <= 0: return 1.0
        if max_wallclock_ms is None:
            ws = int(args.iterations * (1.0 - args.warmdown_fraction))
            return max((args.iterations - step) / max(args.iterations * args.warmdown_fraction, 1), 0.0) if step >= ws else 1.0
        warmdown_ms = max_wallclock_ms * args.warmdown_fraction
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    active_seq_len = args.seq_len_start if args.seq_len_start > 0 else args.train_seq_len
    active_batch_tokens = args.batch_tokens_start if args.batch_tokens_start > 0 else args.train_batch_tokens
    _seq_switched = False; _batch_switched = False

    # --- Compiler warmup ---
    if args.warmup_steps > 0:
        _ms = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        _os = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for mi in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = mi == grad_accum_steps - 1
                x, y = train_loader.next_batch(active_batch_tokens, active_seq_len, grad_accum_steps)
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(x, y)
                (loss * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            log0(f"warmup:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(_ms, strict=True)
        for o, s in zip(optimizers, _os): o.load_state_dict(s)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # --- Main training loop ---
    training_time_ms = 0.0
    stop_after_step = None
    _untied = False
    train_loss = torch.zeros((), device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                         val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            tstats = tern_stats(base_model, group_size=args.bitnet_group_size)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms zero_frac:{tstats['zero_frac']:.3f}")
            torch.cuda.synchronize(); t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # --- HESTIA schedule (update tensor buffers, not Python globals) ---
        if args.hestia_enabled:
            if max_wallclock_ms:
                t_frac = min(elapsed_ms / max_wallclock_ms, 1.0)
            else:
                t_frac = step / max(args.iterations - 1, 1)
            _HESTIA_TAU = args.hestia_tau_init * 0.5 * (1 + math.cos(math.pi * t_frac))
            _HESTIA_PRESSURE = min(1.0, (step / max(args.hestia_pressure_ramp * args.iterations, 1)))
            base_model.set_hestia_params(_HESTIA_TAU, _HESTIA_PRESSURE)

        scale = lr_mul(step, elapsed_ms)

        # Sequence/batch scheduling
        if args.seq_len_start > 0 and not _seq_switched:
            should = elapsed_ms >= args.seq_schedule_fraction * max_wallclock_ms if max_wallclock_ms else step >= int(args.iterations * args.seq_schedule_fraction)
            if should:
                active_seq_len = args.train_seq_len; _seq_switched = True
                torch._dynamo.reset()
                train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
                log0(f"step:{step} seq_len_switch:{args.seq_len_start}->{active_seq_len}")
        if args.batch_tokens_start > 0 and not _batch_switched:
            should = elapsed_ms >= args.batch_schedule_fraction * max_wallclock_ms if max_wallclock_ms else step >= int(args.iterations * args.batch_schedule_fraction)
            if should:
                active_batch_tokens = args.train_batch_tokens; _batch_switched = True
                log0(f"step:{step} batch_switch:{args.batch_tokens_start}->{active_batch_tokens}")

        zero_grad_all()
        train_loss.zero_()
        for micro in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = train_loader.next_batch(active_batch_tokens, active_seq_len, grad_accum_steps)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss.add_(loss.detach())
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Untie lm_head
        if args.untie_at_fraction > 0:
            should = not _untied and (elapsed_ms >= args.untie_at_fraction * max_wallclock_ms if max_wallclock_ms else step >= int(args.iterations * args.untie_at_fraction))
            if should and base_model.tie_embeddings:
                with torch.no_grad():
                    bw = base_model.tok_emb.weight.float()
                    if base_model.lm_head_correction is not None: bw = bw + base_model.lm_head_correction.float()
                    fw = base_model.embed_proj_rev(bw) if base_model.embed_proj_rev is not None else bw
                    base_model.lm_head.weight.copy_(fw)
                base_model.tie_embeddings = False
                base_model.lm_head.weight.requires_grad_(True)
                for g in opt_head.param_groups: g["lr"] = g["base_lr"] = args.head_lr
                _untied = True; torch._dynamo.reset()
                log0(f"step:{step} untied lm_head")

        # Muon momentum warmup
        if args.matrix_optimizer != "adam":
            frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
            for g in opt_muon.param_groups:
                g["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum

        for opt in optimizers:
            for g in opt.param_groups: g["lr"] = g["base_lr"] * scale
            opt.step()
        zero_grad_all()
        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if args.train_log_every > 0 and step % args.train_log_every == 0:
            hestia_info = f" tau:{_HESTIA_TAU:.4f} p:{_HESTIA_PRESSURE:.2f}" if args.hestia_enabled else ""
            log0(f"step:{step}/{args.iterations} loss:{train_loss.item():.4f} t:{approx_ms:.0f}ms "
                 f"avg:{approx_ms/step:.1f}ms{hestia_info}")
        if args.churn_log_every > 0 and step % args.churn_log_every == 0:
            log0(f"step:{step} churn:{churn_fn(base_model, args.bitnet_group_size):.4f} "
                 f"zero:{tern_stats(base_model, args.bitnet_group_size)['zero_frac']:.3f}")

        if stop_after_step is None and max_wallclock_ms is not None and step % 10 == 0:
            reached_cap = approx_ms >= max_wallclock_ms
            if distributed:
                cap_t = torch.tensor(int(reached_cap), device=device)
                dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
                reached_cap = bool(cap_t.item())
            if reached_cap: stop_after_step = step

    # --- Post-training: GPTQ-Ternary Quantization ---
    gptq_hessians = None
    if args.gptq_ternary and master_process:
        log0("--- GPTQ-Ternary: Generating calibration data ---")
        t_gptq = time.perf_counter()
        calib_tokens = base_model.generate(
            n_seqs=args.gptq_calib_seqs, max_len=args.gptq_calib_len,
            temperature=args.gptq_calib_temp, device=device)
        log0(f"  Generated {calib_tokens.shape[0]}x{calib_tokens.shape[1]} calibration tokens")

        log0("--- GPTQ-Ternary: Collecting activations ---")
        gptq_hessians_raw = collect_activations_for_gptq(base_model, calib_tokens, device)
        # Map module names to state_dict names
        gptq_hessians = {}
        for mname, H in gptq_hessians_raw.items():
            gptq_hessians[mname + ".weight"] = H
        gptq_time = time.perf_counter() - t_gptq
        log0(f"  GPTQ calibration: {gptq_time:.1f}s, {len(gptq_hessians)} layers")

    # --- Serialization ---
    if master_process:
        sd = base_model.state_dict()
        if base_model.tie_embeddings:
            sd.pop("lm_head.weight", None)

        # Try both 2:4 and standard packing, pick smaller
        methods = {}
        for use_24 in ([True, False] if args.structured_2_4 else [False]):
            label = "2:4" if use_24 else "base3"
            q_obj, stats = q_sd(sd, group_size=args.bitnet_group_size,
                               fp_storage=args.fp_storage, use_2_4=use_24,
                               gptq_hessians=gptq_hessians)
            buf = io.BytesIO()
            torch.save(q_obj, buf)
            blob = lzma.compress(buf.getvalue(), preset=9)
            methods[label] = {"blob": blob, "stats": stats}
            log0(f"  {label}: {len(blob)/1e6:.2f}MB ternary:{stats['ternary_params']}({stats['ternary_bytes']}B) "
                 f"fp:{stats['fp_params']}({stats['fp_bytes']}B)")

        best = min(methods, key=lambda m: len(methods[m]["blob"]))
        final_blob = methods[best]["blob"]
        q_stats = methods[best]["stats"]

        with open("final_model.hestia.ptz", "wb") as f:
            f.write(final_blob)

        artifact_bytes = len(final_blob)
        code_bytes = len(code.encode("utf-8"))
        total = artifact_bytes + code_bytes
        log0(f"best_method:{best} artifact:{artifact_bytes/1e6:.2f}MB code:{code_bytes}")
        log0(f"budget:{total}/{16000000} ({total/1e6:.2f}/{16.00:.2f}MB) {'FITS' if total <= 16000000 else 'OVER'}")

    # --- Roundtrip validation ---
    if distributed: dist.barrier()
    with open("final_model.hestia.ptz", "rb") as f:
        loaded = torch.load(io.BytesIO(lzma.decompress(f.read())), map_location="cpu", weights_only=False)
    base_model.load_state_dict(deq_sd(loaded), strict=False)
    torch._dynamo.reset()

    q_val_loss, q_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                     val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log0(f"final_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")

    # --- EGGROLL refinement (eval-time budget) ---
    if args.eggroll_enabled and master_process:
        log0("--- EGGROLL Ternary Refinement ---")
        n_flips, eggroll_loss = eggroll_refine(
            base_model, val_tokens, device,
            max_seconds=args.eggroll_seconds,
            candidates_per_iter=args.eggroll_candidates)
        log0(f"  EGGROLL: {n_flips} flips, loss={eggroll_loss:.6f}")

    # --- Temperature scaling ---
    opt_temp = 1.0
    if args.temp_scaling:
        torch.cuda.synchronize(); t_temp = time.perf_counter()
        calib_tok = train_loader.stream.take(65536).to(device)
        opt_temp = find_temp(args, base_model, rank, world_size, device, grad_accum_steps,
                             calib_tok, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
        temp_ms = 1000.0 * (time.perf_counter() - t_temp)
        log0(f"temp_scaling T:{opt_temp:.2f} time:{temp_ms:.0f}ms")

    # --- Sliding window eval ---
    if args.sliding_eval:
        torch.cuda.synchronize(); t_sw = time.perf_counter()
        sw_loss, sw_bpb = eval_val_sliding(args, base_model, rank, world_size, device, grad_accum_steps,
                                           val_tokens, base_bytes_lut, has_leading_space_lut,
                                           is_boundary_token_lut, stride=args.sliding_eval_stride,
                                           temperature=opt_temp)
        sw_ms = 1000.0 * (time.perf_counter() - t_sw)
        log0(f"final_sliding val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} "
             f"(stride={args.sliding_eval_stride}, T={opt_temp:.2f}) time:{sw_ms:.0f}ms")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
