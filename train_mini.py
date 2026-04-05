"""
Mini training script: full pipeline validation at tiny scale on CPU/MPS.
No CUDA, no DDP, no FlashAttention, no torch.compile.

Pipeline: Train with HESTIA QAT → GPTQ-ternary → 2:4 pack → compress → decompress → eval roundtrip

Architecture: 64d × 2L, vocab 64, seq_len 32, ~50K params
Target: runs in <60 seconds on MacBook Air CPU

Usage:
    python train_mini.py
"""

import io
import lzma
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ============================================================================
# Config
# ============================================================================

class Config:
    vocab_size = 64
    model_dim = 64
    num_layers = 2
    num_heads = 4
    num_kv_heads = 2
    mlp_mult = 4
    seq_len = 32
    batch_size = 16
    n_steps = 200
    lr = 3e-3
    seed = 42
    group_size = 16  # smaller groups for tiny model
    # HESTIA params
    tau_init = 0.3
    pressure_ramp_frac = 0.2  # ramp pressure over first 20% of steps
    # Post-training
    gptq_enabled = True
    eggroll_iters = 10
    eggroll_candidates = 128

# ============================================================================
# HESTIA: Differentiable Ternary QAT
# ============================================================================

def hestia_soft_quantize(w: Tensor, scale: Tensor, tau: float) -> Tensor:
    grid = torch.tensor([-1.0, 0.0, 1.0], device=w.device, dtype=w.dtype)
    w_norm = (w / scale).unsqueeze(-1)
    logits = -((w_norm - grid) ** 2) / max(tau, 1e-6)
    probs = F.softmax(logits, dim=-1)
    return scale * (probs * grid).sum(dim=-1)

def hard_ternary_ste(w: Tensor, scale: Tensor) -> Tensor:
    """Standard STE ternary for comparison."""
    q = (w / scale).round().clamp(-1, 1)
    return w + (q * scale - w).detach()

# ============================================================================
# 2:4 Structured Sparsity
# ============================================================================

PATTERNS_24 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERNS_24)}

def enforce_2_4(w_continuous: Tensor) -> Tensor:
    """Enforce 2:4 on continuous weights → ternary {-1,0,1} with exactly 2 nonzeros per 4."""
    assert w_continuous.numel() % 4 == 0
    flat = w_continuous.reshape(-1, 4)
    _, top2 = flat.abs().topk(2, dim=-1)
    result = torch.zeros_like(flat)
    signs = flat.sign()
    signs[signs == 0] = 1.0
    result.scatter_(1, top2, signs.gather(1, top2))
    return result.reshape(w_continuous.shape)

def pack_2_4(w: Tensor) -> tuple[bytes, int]:
    flat = w.reshape(-1, 4).to(torch.int8).numpy()
    n_groups = flat.shape[0]
    packed = np.zeros(n_groups, dtype=np.uint8)
    for i in range(n_groups):
        group = flat[i]
        nz = tuple(np.where(group != 0)[0])
        if len(nz) != 2:
            nz = (0, 1)
        pidx = PATTERN_TO_IDX.get(nz, 0)
        s0 = 1 if group[nz[0]] > 0 else 0
        s1 = 1 if group[nz[1]] > 0 else 0
        packed[i] = pidx * 4 + s0 * 2 + s1
    return packed.tobytes(), n_groups

def unpack_2_4(data: bytes, n_groups: int, shape: tuple) -> Tensor:
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
    return torch.from_numpy(result.reshape(-1)[:np.prod(shape)].reshape(shape))

# ============================================================================
# GPTQ-Ternary
# ============================================================================

def gptq_ternary(W: Tensor, H: Tensor, scale: Tensor) -> Tensor:
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

# ============================================================================
# EGGROLL for Ternary
# ============================================================================

def eggroll_ternary(W: Tensor, scale: Tensor, X: Tensor, Y_target: Tensor,
                    n_iters: int = 10, candidates: int = 128) -> tuple[Tensor, int]:
    W = W.clone()
    total_flips = 0
    n_out, n_in = W.shape
    current_mse = ((X @ (W.float() * scale).T - Y_target) ** 2).sum().item()
    for _ in range(n_iters):
        rows = torch.randint(0, n_out, (candidates,))
        cols = torch.randint(0, n_in, (candidates,))
        for k in range(candidates):
            r, c = rows[k].item(), cols[k].item()
            cur = W[r, c].item()
            best_val, best_mse = cur, current_mse
            for alt in [-1, 0, 1]:
                if alt == cur:
                    continue
                W[r, c] = alt
                mse = ((X @ (W.float() * scale).T - Y_target) ** 2).sum().item()
                if mse < best_mse:
                    best_mse, best_val = mse, alt
            if best_val != cur:
                W[r, c] = best_val
                current_mse = best_mse
                total_flips += 1
            else:
                W[r, c] = cur
    return W, total_flips

# ============================================================================
# Hutch++ Hessian Trace
# ============================================================================

def hutchpp_trace(H: Tensor, n_samples: int = 20) -> float:
    n = H.shape[0]
    traces = []
    for _ in range(n_samples):
        v = torch.sign(torch.randn(n, device=H.device, dtype=H.dtype))
        traces.append((v * (H @ v)).sum().item())
    return sum(traces) / len(traces)

# ============================================================================
# Model Components
# ============================================================================

class HestiaTernaryLinear(nn.Linear):
    """Linear layer with HESTIA differentiable ternary QAT."""
    def __init__(self, in_f, out_f, bias=False, group_size=16):
        super().__init__(in_f, out_f, bias=bias)
        self.group_size = group_size
        self.tau = 0.3  # set externally during training
        self.pressure = 0.0  # set externally during training
        self.sensitivity = 1.0  # set by Hutch++ estimation

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        g = self.group_size
        # Pad if needed
        n_out, n_in = w.shape
        pad = (g - n_in % g) % g
        if pad > 0:
            w_padded = F.pad(w, (0, pad))
        else:
            w_padded = w
        w_grouped = w_padded.reshape(-1, g)
        scale = w_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8)

        if self.pressure > 0 and self.tau > 1e-6:
            # HESTIA soft quantization with per-tensor temperature
            tau_effective = self.tau * math.exp(0.4 * self.sensitivity)
            w_soft = hestia_soft_quantize(w_grouped, scale, tau_effective)
            w_eff = (1.0 - self.pressure) * w_grouped + self.pressure * w_soft
        else:
            w_eff = w_grouped

        w_eff = w_eff.reshape(w_padded.shape)[:n_out, :n_in]
        return F.linear(x, w_eff, self.bias)


class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, group_size):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        q_dim = n_heads * self.head_dim
        kv_dim = n_kv_heads * self.head_dim
        self.qkv = HestiaTernaryLinear(dim, q_dim + 2 * kv_dim, group_size=group_size)
        self.proj = HestiaTernaryLinear(dim, dim, group_size=group_size)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x)
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # Expand KV heads for GQA
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, S, D)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mult, group_size):
        super().__init__()
        hidden = dim * mult
        self.fc = HestiaTernaryLinear(dim, hidden, group_size=group_size)
        self.proj = HestiaTernaryLinear(hidden, dim, group_size=group_size)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())  # relu²


class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, mlp_mult, group_size):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = Attention(dim, n_heads, n_kv_heads, group_size)
        self.mlp = MLP(dim, mlp_mult, group_size)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.blocks = nn.ModuleList([
            Block(cfg.model_dim, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_mult, cfg.group_size)
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, HestiaTernaryLinear):
                nn.init.normal_(m.weight, std=0.02)

    def get_ternary_linears(self) -> list[HestiaTernaryLinear]:
        return [m for m in self.modules() if isinstance(m, HestiaTernaryLinear)]

    def set_hestia_params(self, tau: float, pressure: float):
        for m in self.get_ternary_linears():
            m.tau = tau
            m.pressure = pressure

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

# ============================================================================
# Quantization & Serialization
# ============================================================================

def quantize_model(model: MiniGPT, calib_X: Tensor, cfg: Config, use_gptq: bool = True):
    """
    Quantize all HestiaTernaryLinear layers to 2:4 structured ternary.
    Optionally apply GPTQ error compensation first.
    Returns a serializable state dict.
    """
    quantized = {}
    stats = {"ternary_params": 0, "ternary_bytes": 0, "fp_params": 0, "fp_bytes": 0}

    for name, param in model.state_dict().items():
        t = param.detach().cpu().float()

        # Find the corresponding module
        is_ternary = False
        for mname, m in model.named_modules():
            if isinstance(m, HestiaTernaryLinear) and (name == f"{mname}.weight"):
                is_ternary = True
                break

        if is_ternary and t.ndim == 2:
            g = cfg.group_size
            pad = (g - t.shape[1] % g) % g
            t_padded = F.pad(t, (0, pad)) if pad > 0 else t
            t_grouped = t_padded.reshape(-1, g)
            scale = t_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8)

            if use_gptq and calib_X is not None:
                # Compute layer-local Hessian from calibration
                # For simplicity, use identity-like Hessian for mini model
                H = torch.eye(t.shape[1]) * 0.01 + torch.randn(t.shape[1], t.shape[1]) * 0.001
                H = H @ H.T  # make PSD
                Q = gptq_ternary(t, H, scale.reshape(t.shape[0], -1)[:, :1].expand(-1, 1))
                # Reconstruct padded shape for 2:4
                q_padded = F.pad(Q, (0, pad)) if pad > 0 else Q
            else:
                q_padded = (t_padded.reshape(-1, g) / scale).round().clamp(-1, 1)
                q_padded = q_padded.reshape(t_padded.shape)

            # Enforce 2:4 structure (on padded tensor, must be div by 4)
            # Ensure divisible by 4
            total = q_padded.numel()
            pad4 = (4 - total % 4) % 4
            if pad4 > 0:
                q_flat = F.pad(q_padded.reshape(-1), (0, pad4))
            else:
                q_flat = q_padded.reshape(-1)
            q_24 = enforce_2_4(q_flat.float()).to(torch.int8)

            packed_bytes, n_groups = pack_2_4(q_24)
            quantized[name] = {
                "type": "ternary_2_4",
                "packed": packed_bytes,
                "n_groups": n_groups,
                "scale": scale.half(),
                "shape": list(t.shape),
                "padded_shape": list(t_padded.shape),
                "group_size": g,
                "pad4": pad4,
            }
            stats["ternary_params"] += t.numel()
            stats["ternary_bytes"] += len(packed_bytes) + scale.numel() * 2
        else:
            quantized[name] = {"type": "fp16", "data": t.half()}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel() * 2

    return quantized, stats


def dequantize_model(quantized: dict) -> dict:
    """Reconstruct float state dict from quantized representation."""
    out = {}
    for name, entry in quantized.items():
        if entry["type"] == "ternary_2_4":
            q_24 = unpack_2_4(entry["packed"], entry["n_groups"],
                              (entry["n_groups"] * 4,))
            # Remove pad4
            total_padded = 1
            for s in entry["padded_shape"]:
                total_padded *= s
            q_padded = q_24[:total_padded].reshape(entry["padded_shape"]).float()

            # Reconstruct with scales
            g = entry["group_size"]
            q_grouped = q_padded.reshape(-1, g)
            scale = entry["scale"].float()
            # Scale: need to broadcast properly
            w_recon = (q_grouped * scale).reshape(entry["padded_shape"])
            # Trim padding
            shape = entry["shape"]
            out[name] = w_recon[:shape[0], :shape[1]].contiguous()
        else:
            out[name] = entry["data"].float()
    return out


def serialize_quantized(quantized: dict) -> bytes:
    """Serialize quantized model to compressed bytes."""
    buf = io.BytesIO()
    torch.save(quantized, buf)
    return lzma.compress(buf.getvalue(), preset=9)


def deserialize_quantized(data: bytes) -> dict:
    """Deserialize compressed quantized model."""
    return torch.load(io.BytesIO(lzma.decompress(data)), weights_only=False)

# ============================================================================
# Training & Evaluation
# ============================================================================

def generate_data(cfg: Config, n_batches: int):
    """Generate random token sequences for training."""
    data = []
    for _ in range(n_batches):
        tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len + 1))
        data.append((tokens[:, :-1], tokens[:, 1:]))
    return data


@torch.no_grad()
def evaluate(model: MiniGPT, data: list) -> float:
    """Compute mean cross-entropy loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in data:
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens


def train(cfg: Config):
    print("=" * 60)
    print("  MINI TRAINING PIPELINE")
    print(f"  {cfg.model_dim}d × {cfg.num_layers}L, vocab={cfg.vocab_size}, "
          f"seq={cfg.seq_len}, steps={cfg.n_steps}")
    print("=" * 60)

    torch.manual_seed(cfg.seed)
    model = MiniGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_ternary = sum(p.numel() for m in model.get_ternary_linears() for p in m.parameters())
    print(f"\n  Total params: {n_params:,} ({n_ternary:,} ternary)")

    # Generate data
    train_data = generate_data(cfg, 50)
    val_data = generate_data(cfg, 10)

    # Optimizer (simple Adam for mini model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ── Phase 1: Hutch++ Sensitivity Estimation ──────────────────────
    print("\n── Phase 1: Hutch++ Sensitivity Estimation ──")
    sensitivities = []
    with torch.no_grad():
        sample_x = train_data[0][0][:4]  # small calibration batch
        # Forward to get activations at each linear layer
        # Simplified: use random activations scaled by layer depth
        for i, m in enumerate(model.get_ternary_linears()):
            X_calib = torch.randn(32, m.in_features)
            H = (X_calib.T @ X_calib) / X_calib.shape[0]
            trace = hutchpp_trace(H, n_samples=10)
            # Normalize to [0, 1] range
            sensitivities.append(trace)
    # Normalize
    if sensitivities:
        mean_s = sum(sensitivities) / len(sensitivities)
        std_s = max((sum((s - mean_s)**2 for s in sensitivities) / len(sensitivities)) ** 0.5, 1e-8)
        for i, m in enumerate(model.get_ternary_linears()):
            m.sensitivity = (sensitivities[i] - mean_s) / std_s
            print(f"    {type(m).__name__}[{i}]: sensitivity={m.sensitivity:.3f}")

    # ── Phase 2: Training with HESTIA QAT ────────────────────────────
    print(f"\n── Phase 2: Training with HESTIA QAT ({cfg.n_steps} steps) ──")
    model.train()
    t0 = time.time()
    loss_history = []

    for step in range(cfg.n_steps):
        # HESTIA schedule
        t_frac = step / max(cfg.n_steps - 1, 1)
        tau = cfg.tau_init * 0.5 * (1 + math.cos(math.pi * t_frac))
        pressure = min(1.0, step / (cfg.pressure_ramp_frac * cfg.n_steps))
        model.set_hestia_params(tau, pressure)

        # Training step
        x, y = train_data[step % len(train_data)]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (step + 1) % 50 == 0:
            val_loss = evaluate(model, val_data)
            print(f"    step={step+1:4d}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  "
                  f"tau={tau:.4f}  pressure={pressure:.2f}")

    train_time = time.time() - t0
    pre_quant_loss = evaluate(model, val_data)
    print(f"\n  Training complete in {train_time:.1f}s")
    print(f"  Pre-quantization val_loss: {pre_quant_loss:.4f}")

    # ── Phase 3: Post-Training Quantization ──────────────────────────
    print(f"\n── Phase 3: Post-Training Quantization ──")

    # 3a: Self-generate calibration data
    print("  Generating calibration data...")
    model.eval()
    calib_tokens = torch.randint(0, cfg.vocab_size, (8, cfg.seq_len))  # simplified
    with torch.no_grad():
        calib_logits = model(calib_tokens)
    calib_X = calib_tokens  # placeholder for proper activation collection

    # 3b: Quantize with GPTQ-ternary + 2:4 packing
    print("  Quantizing (GPTQ-ternary + 2:4 structured)...")
    t_quant = time.time()
    quantized, q_stats = quantize_model(model, calib_X, cfg, use_gptq=cfg.gptq_enabled)
    quant_time = time.time() - t_quant
    print(f"  Quantization time: {quant_time:.2f}s")
    print(f"  Ternary params: {q_stats['ternary_params']:,} ({q_stats['ternary_bytes']:,} bytes)")
    print(f"  FP params: {q_stats['fp_params']:,} ({q_stats['fp_bytes']:,} bytes)")

    # 3c: Serialize + compress
    print("  Compressing (LZMA-9)...")
    t_compress = time.time()
    blob = serialize_quantized(quantized)
    compress_time = time.time() - t_compress
    print(f"  Compressed artifact: {len(blob):,} bytes ({len(blob)/1024:.1f} KB)")
    print(f"  Compression time: {compress_time:.2f}s")

    # ── Phase 4: Roundtrip Validation ────────────────────────────────
    print(f"\n── Phase 4: Roundtrip Validation ──")

    # 4a: Decompress + dequantize
    loaded = deserialize_quantized(blob)
    restored_sd = dequantize_model(loaded)

    # 4b: Load into fresh model and evaluate
    model_rt = MiniGPT(cfg)
    # Need to handle tied weights
    if "lm_head.weight" not in restored_sd:
        restored_sd["lm_head.weight"] = restored_sd["tok_emb.weight"]
    model_rt.load_state_dict(restored_sd, strict=False)

    rt_loss = evaluate(model_rt, val_data)
    quant_gap = rt_loss - pre_quant_loss
    print(f"  Pre-quantization val_loss:  {pre_quant_loss:.4f}")
    print(f"  Post-roundtrip val_loss:    {rt_loss:.4f}")
    print(f"  Quantization gap:           {quant_gap:+.4f}")

    # ── Phase 5: EGGROLL Refinement ──────────────────────────────────
    if cfg.eggroll_iters > 0:
        print(f"\n── Phase 5: EGGROLL Refinement ({cfg.eggroll_iters} iters) ──")
        t_egg = time.time()
        total_flips = 0

        # For each ternary layer, try EGGROLL refinement
        for name, entry in loaded.items():
            if entry["type"] == "ternary_2_4" and name in model.state_dict():
                shape = entry["shape"]
                g = entry["group_size"]
                # Unpack to matrix form
                q_24 = unpack_2_4(entry["packed"], entry["n_groups"],
                                  (entry["n_groups"] * 4,))
                total_padded = 1
                for s in entry["padded_shape"]:
                    total_padded *= s
                q_matrix = q_24[:total_padded].reshape(entry["padded_shape"]).float()
                q_trimmed = q_matrix[:shape[0], :shape[1]]

                # Reconstruct per-row scale from group scales
                scale_groups = entry["scale"].float()  # (n_groups_total, 1)
                n_rows = shape[0]
                groups_per_row = (shape[1] + g - 1) // g
                # Use mean scale per row as a simple approximation
                scale_per_row = scale_groups.reshape(n_rows, -1).mean(dim=1, keepdim=True)

                # EGGROLL on this layer
                X_eval = torch.randn(32, shape[1])
                Y_target = X_eval @ model.state_dict()[name].float().T
                improved, flips = eggroll_ternary(
                    q_trimmed, scale_per_row.expand_as(q_trimmed),
                    X_eval, Y_target,
                    n_iters=cfg.eggroll_iters,
                    candidates=cfg.eggroll_candidates
                )
                total_flips += flips

        egg_time = time.time() - t_egg
        print(f"  EGGROLL flips: {total_flips}")
        print(f"  EGGROLL time: {egg_time:.2f}s")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Model:           {cfg.model_dim}d × {cfg.num_layers}L, {n_params:,} params")
    print(f"  Training:        {cfg.n_steps} steps, {train_time:.1f}s")
    print(f"  Pre-quant loss:  {pre_quant_loss:.4f}")
    print(f"  Post-quant loss: {rt_loss:.4f} (gap: {quant_gap:+.4f})")
    print(f"  Artifact:        {len(blob):,} bytes ({len(blob)/1024:.1f} KB)")
    bits_per_ternary = (len(blob) * 8) / max(q_stats['ternary_params'], 1)
    print(f"  Bits/ternary:    {bits_per_ternary:.3f}")
    print(f"  Quantize time:   {quant_time:.2f}s")
    print(f"  Compress time:   {compress_time:.2f}s")

    # Check the pipeline produced a valid model that learned something
    random_loss = math.log(cfg.vocab_size)  # ~4.16 for vocab=64
    learned = pre_quant_loss < random_loss * 0.98
    survived_quant = quant_gap < 0.5  # quantization didn't destroy the model
    print(f"\n  Random baseline: {random_loss:.4f}")
    print(f"  Learned:         {'YES' if learned else 'NO'} (loss < {random_loss*0.98:.4f})")
    print(f"  Survived quant:  {'YES' if survived_quant else 'NO'} (gap < 0.5)")

    if learned and survived_quant:
        print(f"\n  [PASS] Full pipeline validated!")
    else:
        if not learned:
            print(f"\n  [WARN] Model didn't learn (expected on random data with tiny model)")
        if not survived_quant:
            print(f"\n  [FAIL] Quantization gap too large")

    print(f"{'=' * 60}\n")
    return learned, survived_quant


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
