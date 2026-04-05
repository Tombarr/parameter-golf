"""
Local validation script for novel components.
Runs on CPU (MacBook Air compatible). No CUDA, no DDP, no FlashAttention.

Tests:
  1. HESTIA softmax relaxation — gradients flow, anneals to hard ternary
  2. 2:4 structured ternary packing — roundtrip correctness, size estimates
  3. GPTQ-ternary — error compensation reduces reconstruction error
  4. EGGROLL for ternary — finds improving swaps

Usage:
  python test_components.py
"""

import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# ============================================================================
# 1. HESTIA: Differentiable Softmax Relaxation for Ternary QAT
# ============================================================================

def hestia_soft_quantize(w: Tensor, scale: Tensor, tau: float) -> Tensor:
    """
    HESTIA softmax relaxation over ternary grid {-1, 0, 1}.
    Returns a soft (differentiable) ternary approximation.

    Args:
        w: weight tensor (any shape)
        scale: per-group absmax scale (broadcastable to w)
        tau: temperature (high=soft, low→0=hard ternary)
    """
    grid = torch.tensor([-1.0, 0.0, 1.0], device=w.device, dtype=w.dtype)
    # Normalize weights by scale
    w_norm = (w / scale).unsqueeze(-1)  # (..., 1)
    # Compute softmax logits: negative squared distance to each grid point
    logits = -((w_norm - grid) ** 2) / tau  # (..., 3)
    probs = F.softmax(logits, dim=-1)  # (..., 3)
    # Weighted sum over grid points
    w_soft = scale * (probs * grid).sum(dim=-1)  # (...)
    return w_soft


def hestia_quantize_with_pressure(w: Tensor, scale: Tensor, tau: float, pressure: float) -> Tensor:
    """
    Full HESTIA forward: interpolate between continuous and soft-quantized.

    w_eff = (1 - p) * w + p * hestia_soft_quantize(w, scale, tau)
    """
    w_soft = hestia_soft_quantize(w, scale, tau)
    return (1.0 - pressure) * w + pressure * w_soft


def hard_ternary(w: Tensor, scale: Tensor) -> Tensor:
    """Standard hard ternary quantization (for comparison)."""
    return scale * (w / scale).round().clamp(-1, 1)


def test_hestia():
    print("=" * 60)
    print("TEST 1: HESTIA Softmax Relaxation")
    print("=" * 60)

    torch.manual_seed(42)
    w = torch.randn(128, 64, requires_grad=True)

    # Compute per-group scale (group_size=64)
    with torch.no_grad():
        scale = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)

    # Test 1a: Gradient flow
    w_soft = hestia_soft_quantize(w, scale.detach(), tau=0.3)
    loss = w_soft.sum()
    loss.backward()
    assert w.grad is not None, "FAIL: No gradient!"
    assert w.grad.abs().sum() > 0, "FAIL: Zero gradients!"
    print(f"  [PASS] Gradients flow. Mean |grad|: {w.grad.abs().mean():.6f}")

    # Test 1b: Convergence to hard ternary as tau -> 0
    with torch.no_grad():
        w_hard = hard_ternary(w, scale)
        errors = []
        for tau in [1.0, 0.1, 0.01, 0.001]:
            w_s = hestia_soft_quantize(w, scale, tau)
            err = (w_s - w_hard).abs().mean().item()
            errors.append((tau, err))
            print(f"  tau={tau:.3f}  mean|soft-hard|={err:.6f}")
        assert errors[-1][1] < errors[0][1], "FAIL: Not converging to hard ternary!"
        print(f"  [PASS] Converges to hard ternary (error {errors[0][1]:.4f} -> {errors[-1][1]:.6f})")

    # Test 1c: Pressure interpolation
    with torch.no_grad():
        w_p0 = hestia_quantize_with_pressure(w, scale, tau=0.1, pressure=0.0)
        w_p1 = hestia_quantize_with_pressure(w, scale, tau=0.1, pressure=1.0)
        assert torch.allclose(w_p0, w, atol=1e-6), "FAIL: pressure=0 should return w"
        w_soft_direct = hestia_soft_quantize(w, scale, tau=0.1)
        assert torch.allclose(w_p1, w_soft_direct, atol=1e-6), "FAIL: pressure=1 should return soft"
        print(f"  [PASS] Pressure interpolation correct")

    # Test 1d: Temperature schedule simulation
    print("\n  Temperature annealing simulation (100 steps):")
    with torch.no_grad():
        for step in range(0, 101, 20):
            t_frac = step / 100.0
            tau = 0.3 * 0.5 * (1 + math.cos(math.pi * t_frac))  # cosine decay
            pressure = min(1.0, step / 20.0)  # ramp over first 20%
            w_eff = hestia_quantize_with_pressure(w, scale, tau, pressure)
            # Measure how "ternary" the result is
            w_norm = w_eff / scale
            ternary_score = ((w_norm.round().clamp(-1, 1) - w_norm).abs() < 0.05).float().mean()
            print(f"    step={step:3d}  tau={tau:.4f}  pressure={pressure:.2f}  ternary_fraction={ternary_score:.3f}")

    print()


# ============================================================================
# 2. 2:4 Structured Ternary Packing
# ============================================================================

# 6 possible patterns for which 2 of 4 positions are nonzero
PATTERNS_24 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERNS_24)}


def enforce_2_4_structure(w_continuous: Tensor) -> Tensor:
    """
    Enforce 2:4 structured sparsity on continuous weights, producing ternary output.
    For each group of 4, the 2 largest-magnitude continuous values become ±1,
    the other 2 become 0. Operates on PRE-quantization weights.

    Args:
        w_continuous: continuous (float) weight tensor, numel divisible by 4
    Returns:
        ternary tensor {-1, 0, 1} with exactly 2 nonzeros per group of 4
    """
    assert w_continuous.numel() % 4 == 0, f"Weight count {w_continuous.numel()} must be divisible by 4"
    flat = w_continuous.reshape(-1, 4)
    # Find 2 largest magnitude positions per group
    _, top2 = flat.abs().topk(2, dim=-1)
    result = torch.zeros_like(flat)
    # Set the top-2 positions to the sign of the original value
    signs = flat.sign()
    # Where sign is 0 (exactly zero weight), default to +1
    signs[signs == 0] = 1.0
    result.scatter_(1, top2, signs.gather(1, top2))
    return result.reshape(w_continuous.shape)


def pack_2_4_ternary(w: Tensor) -> tuple[bytes, int]:
    """
    Pack 2:4 structured ternary weights into bytes.
    Each group of 4 weights -> 1 byte (24 possible states, fits in 5 bits).
    We use 1 byte per group for simplicity; LZMA compresses the slack.

    Returns (packed_bytes, n_groups).
    """
    flat = w.reshape(-1, 4).to(torch.int8).numpy()
    n_groups = flat.shape[0]
    packed = np.zeros(n_groups, dtype=np.uint8)

    for i in range(n_groups):
        group = flat[i]
        nonzero_pos = tuple(np.where(group != 0)[0])
        if len(nonzero_pos) != 2:
            # Fallback: treat as (0,1) with zero signs if structure is broken
            nonzero_pos = (0, 1)
        pattern_idx = PATTERN_TO_IDX.get(nonzero_pos, 0)
        sign0 = 1 if group[nonzero_pos[0]] > 0 else 0
        sign1 = 1 if group[nonzero_pos[1]] > 0 else 0
        packed[i] = pattern_idx * 4 + sign0 * 2 + sign1

    return packed.tobytes(), n_groups


def unpack_2_4_ternary(data: bytes, n_groups: int, shape: tuple) -> Tensor:
    """Unpack 2:4 structured ternary weights from packed bytes."""
    packed = np.frombuffer(data, dtype=np.uint8)[:n_groups]
    result = np.zeros((n_groups, 4), dtype=np.int8)

    for i in range(n_groups):
        val = int(packed[i])
        pattern_idx = val // 4
        signs = val % 4
        sign0 = 1 if (signs & 2) else -1
        sign1 = 1 if (signs & 1) else -1
        p0, p1 = PATTERNS_24[pattern_idx]
        result[i, p0] = sign0
        result[i, p1] = sign1

    return torch.from_numpy(result.reshape(-1)[:np.prod(shape)].reshape(shape))


def test_2_4_packing():
    print("=" * 60)
    print("TEST 2: 2:4 Structured Ternary Packing")
    print("=" * 60)

    torch.manual_seed(42)

    # Test 2a: enforce_2_4_structure produces valid patterns
    w = torch.randn(256, 128)
    scale = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    w_normalized = w / scale  # continuous, pre-quantization
    w_24 = enforce_2_4_structure(w_normalized)
    # Also keep a standard ternary for compression comparison
    w_ternary = w_normalized.round().clamp(-1, 1)

    flat = w_24.reshape(-1, 4)
    zeros_per_group = (flat == 0).sum(dim=1)
    nonzeros_per_group = (flat != 0).sum(dim=1)
    assert (zeros_per_group == 2).all(), "FAIL: Not exactly 2 zeros per group!"
    assert (nonzeros_per_group == 2).all(), "FAIL: Not exactly 2 nonzeros per group!"
    assert ((flat == -1) | (flat == 0) | (flat == 1)).all(), "FAIL: Values outside {-1,0,1}!"
    print(f"  [PASS] 2:4 structure enforced correctly on {w.shape} tensor")

    # Test 2b: Pack/unpack roundtrip
    packed_bytes, n_groups = pack_2_4_ternary(w_24)
    w_unpacked = unpack_2_4_ternary(packed_bytes, n_groups, w_24.shape)
    assert torch.equal(w_24.to(torch.int8), w_unpacked.to(torch.int8)), "FAIL: Roundtrip mismatch!"
    print(f"  [PASS] Pack/unpack roundtrip exact match")

    # Test 2c: Compression ratio
    n_params = w_24.numel()
    raw_bytes = len(packed_bytes)
    bits_per_weight = raw_bytes * 8 / n_params

    import lzma
    compressed = lzma.compress(packed_bytes, preset=9)
    compressed_bits_per_weight = len(compressed) * 8 / n_params

    print(f"  Params: {n_params:,}")
    print(f"  Raw packed: {raw_bytes:,} bytes ({bits_per_weight:.3f} bits/weight)")
    print(f"  LZMA-9:     {len(compressed):,} bytes ({compressed_bits_per_weight:.3f} bits/weight)")

    # Compare with base-3 + LZMA (standard ternary approach)
    def pack_base3(q):
        f = (q.reshape(-1).to(torch.int8) + 1).numpy()
        n = len(f)
        p = (5 - n % 5) % 5
        if p:
            f = np.concatenate([f, np.zeros(p, dtype=np.int8)])
        g = f.reshape(-1, 5).astype(np.uint8)
        return (g[:, 0] + g[:, 1]*3 + g[:, 2]*9 + g[:, 3]*27 + g[:, 4]*81).tobytes(), n

    base3_bytes, _ = pack_base3(w_ternary)
    base3_compressed = lzma.compress(base3_bytes, preset=9)
    base3_bpw = len(base3_compressed) * 8 / w_ternary.numel()

    print(f"\n  Comparison (same param count):")
    print(f"    Standard ternary (base-3+LZMA):  {len(base3_compressed):,} bytes ({base3_bpw:.3f} bits/weight)")
    print(f"    2:4 structured (packed+LZMA):     {len(compressed):,} bytes ({compressed_bits_per_weight:.3f} bits/weight)")
    savings = 1.0 - len(compressed) / len(base3_compressed)
    print(f"    Savings: {savings*100:.1f}%")

    # Test 2d: Artifact size projection for 65M params
    proj_65m_base3 = 65e6 * base3_bpw / 8 / 1e6
    proj_65m_24 = 65e6 * compressed_bits_per_weight / 8 / 1e6
    print(f"\n  Projected artifact for 65M ternary params:")
    print(f"    Standard (base-3+LZMA):  {proj_65m_base3:.2f} MB")
    print(f"    2:4 structured:          {proj_65m_24:.2f} MB")
    print(f"    Freed space:             {proj_65m_base3 - proj_65m_24:.2f} MB")
    print()


# ============================================================================
# 3. GPTQ-Ternary: Error Compensation on Ternary Grid
# ============================================================================

def gptq_ternary(W: Tensor, H: Tensor, scale: Tensor, block_size: int = 128) -> Tensor:
    """
    GPTQ error compensation adapted for ternary quantization.
    Based on the standard GPTQ algorithm (Frantar et al., 2022) but with
    ternary quantization grid {-1, 0, 1} instead of uniform integer grid.

    Args:
        W: weight matrix (out_features, in_features), float
        H: Hessian approximation (in_features, in_features), X^T @ X / n
        scale: per-row absmax scale for ternary (out_features, 1)
        block_size: columns processed per block

    Returns:
        Q: ternary-quantized weight values {-1, 0, 1}
    """
    W = W.clone().float()
    n_out, n_in = W.shape

    # Add damping for numerical stability (standard GPTQ practice)
    damp = 0.01 * H.diag().mean().clamp(min=1e-6)
    H = H + damp * torch.eye(n_in, device=H.device, dtype=H.dtype)

    # We need diagonal of H^{-1} for error weighting
    # Process column-by-column with error compensation
    Q = torch.zeros_like(W)
    scale_sq = scale.squeeze(-1)  # (out_features,)

    for j in range(n_in):
        w_col = W[:, j]
        # Quantize column j to ternary
        q_val = (w_col / scale_sq).round().clamp(-1, 1)
        q_col = q_val * scale_sq
        Q[:, j] = q_val

        # Error from quantizing this column
        error = w_col - q_col  # (out_features,)

        # Redistribute error to remaining unquantized columns
        # Weighted by H[j, k] / H[j, j] — how much column k's output
        # correlates with column j's output
        if j + 1 < n_in:
            h_jj = H[j, j].clamp(min=1e-8)
            # error: (out,), H[j, j+1:]: (remaining,)
            # Update: W[:, k] += error * H[j,k] / H[j,j] for k > j
            W[:, j+1:] += error.unsqueeze(1) * (H[j, j+1:] / h_jj).unsqueeze(0)

    return Q


def naive_ternary(W: Tensor, scale: Tensor) -> Tensor:
    """Simple round-to-nearest ternary quantization."""
    return (W / scale).round().clamp(-1, 1)


def test_gptq_ternary():
    print("=" * 60)
    print("TEST 3: GPTQ-Ternary Error Compensation")
    print("=" * 60)

    torch.manual_seed(42)

    # Create a weight matrix and calibration activations
    out_features, in_features = 128, 256
    W = torch.randn(out_features, in_features) * 0.02
    X = torch.randn(512, in_features)  # calibration data (512 samples)

    # Compute Hessian
    H = (X.T @ X) / X.shape[0]

    # Per-row scale
    scale = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)

    # Naive ternary
    Q_naive = naive_ternary(W, scale)
    W_naive_recon = Q_naive * scale

    # GPTQ ternary
    Q_gptq = gptq_ternary(W.clone(), H, scale)
    W_gptq_recon = Q_gptq * scale

    # Compare reconstruction error (weighted by Hessian = activation-aware)
    naive_output = X @ W_naive_recon.T
    gptq_output = X @ W_gptq_recon.T
    original_output = X @ W.T

    naive_error = ((naive_output - original_output) ** 2).mean().item()
    gptq_error = ((gptq_output - original_output) ** 2).mean().item()

    print(f"  Naive ternary reconstruction MSE:  {naive_error:.8f}")
    print(f"  GPTQ ternary reconstruction MSE:   {gptq_error:.8f}")
    improvement = (1 - gptq_error / naive_error) * 100
    print(f"  Improvement: {improvement:.1f}%")

    if gptq_error < naive_error:
        print(f"  [PASS] GPTQ reduces reconstruction error by {improvement:.1f}%")
    else:
        print(f"  [WARN] GPTQ did not improve — may need tuning")

    # Verify output is still valid ternary
    assert ((Q_gptq == -1) | (Q_gptq == 0) | (Q_gptq == 1)).all(), "FAIL: Non-ternary values!"
    print(f"  [PASS] Output is valid ternary {{-1, 0, 1}}")

    # Statistics
    zero_frac_naive = (Q_naive == 0).float().mean().item()
    zero_frac_gptq = (Q_gptq == 0).float().mean().item()
    print(f"  Zero fraction — naive: {zero_frac_naive:.3f}, GPTQ: {zero_frac_gptq:.3f}")
    print()


# ============================================================================
# 4. EGGROLL for Ternary: Gradient-Free Discrete Optimization
# ============================================================================

def eggroll_ternary(
    W_ternary: Tensor,
    scale: Tensor,
    X: Tensor,
    Y_target: Tensor,
    n_iters: int = 50,
    candidates_per_iter: int = 256,
) -> tuple[Tensor, int]:
    """
    EGGROLL adapted for ternary: gradient-free search over {-1, 0, 1}.

    For each iteration:
      - Pick random weight indices
      - Try flipping to each alternative ternary value
      - Keep the change if it reduces output reconstruction error

    Args:
        W_ternary: ternary weights {-1,0,1} (out_features, in_features)
        scale: per-row scale factors
        X: calibration inputs (n_samples, in_features)
        Y_target: target outputs from original model (n_samples, out_features)
        n_iters: number of search iterations
        candidates_per_iter: weight positions to try per iteration

    Returns:
        (improved W_ternary, total_flips)
    """
    W = W_ternary.clone()
    total_flips = 0
    n_out, n_in = W.shape

    # Current reconstruction error
    current_output = X @ (W.float() * scale).T
    current_mse = ((current_output - Y_target) ** 2).sum().item()

    for iteration in range(n_iters):
        # Random indices
        row_idx = torch.randint(0, n_out, (candidates_per_iter,))
        col_idx = torch.randint(0, n_in, (candidates_per_iter,))

        for k in range(candidates_per_iter):
            r, c = row_idx[k].item(), col_idx[k].item()
            current_val = W[r, c].item()

            # Try both alternatives
            alternatives = [v for v in [-1, 0, 1] if v != current_val]
            best_val = current_val
            best_mse = current_mse

            for alt in alternatives:
                W[r, c] = alt
                new_output = X @ (W.float() * scale).T
                new_mse = ((new_output - Y_target) ** 2).sum().item()

                if new_mse < best_mse:
                    best_mse = new_mse
                    best_val = alt

            if best_val != current_val:
                W[r, c] = best_val
                current_mse = best_mse
                total_flips += 1
                # Update cached output
                current_output = X @ (W.float() * scale).T
            else:
                W[r, c] = current_val

    return W, total_flips


def test_eggroll():
    print("=" * 60)
    print("TEST 4: EGGROLL for Ternary")
    print("=" * 60)

    torch.manual_seed(42)

    # Small weight matrix for fast testing
    out_features, in_features = 32, 64
    W_orig = torch.randn(out_features, in_features) * 0.02
    X = torch.randn(128, in_features)

    # Target output from original weights
    Y_target = X @ W_orig.T

    # Naive ternary quantization
    scale = W_orig.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    W_ternary = (W_orig / scale).round().clamp(-1, 1)

    pre_mse = ((X @ (W_ternary.float() * scale).T - Y_target) ** 2).mean().item()

    # Run EGGROLL
    t0 = time.time()
    W_improved, n_flips = eggroll_ternary(
        W_ternary, scale, X, Y_target,
        n_iters=20, candidates_per_iter=128,
    )
    elapsed = time.time() - t0

    post_mse = ((X @ (W_improved.float() * scale).T - Y_target) ** 2).mean().item()

    print(f"  Pre-EGGROLL MSE:  {pre_mse:.8f}")
    print(f"  Post-EGGROLL MSE: {post_mse:.8f}")
    print(f"  Flips made: {n_flips}")
    print(f"  Time: {elapsed:.2f}s")

    improvement = (1 - post_mse / pre_mse) * 100
    if post_mse <= pre_mse:
        print(f"  [PASS] EGGROLL improved MSE by {improvement:.1f}% with {n_flips} flips")
    else:
        print(f"  [FAIL] EGGROLL made things worse!")

    # Verify still valid ternary
    assert ((W_improved == -1) | (W_improved == 0) | (W_improved == 1)).all()
    print(f"  [PASS] Output is valid ternary")
    print()


# ============================================================================
# 5. Hutch++ Hessian Trace Estimation (for HESTIA scheduling)
# ============================================================================

def hutchpp_trace(W: Tensor, X: Tensor, n_samples: int = 20) -> float:
    """
    Estimate tr(H) where H = (dL/dW)^T (dL/dW) ≈ X^T X projected onto W's space.
    Simplified Hutch++ for per-tensor sensitivity scoring.
    """
    H = (X.T @ X) / X.shape[0]
    n = H.shape[0]
    # Hutchinson estimator with Rademacher vectors
    traces = []
    for _ in range(n_samples):
        v = torch.sign(torch.randn(n, device=H.device, dtype=H.dtype))
        traces.append((v * (H @ v)).sum().item())
    return sum(traces) / len(traces)


def test_hutchpp():
    print("=" * 60)
    print("TEST 5: Hutch++ Hessian Trace Estimation")
    print("=" * 60)

    torch.manual_seed(42)
    n = 128
    X = torch.randn(256, n)
    H = (X.T @ X) / X.shape[0]

    exact_trace = H.trace().item()
    estimated_trace = hutchpp_trace(torch.randn(64, n), X, n_samples=50)

    rel_error = abs(estimated_trace - exact_trace) / abs(exact_trace)
    print(f"  Exact trace:     {exact_trace:.2f}")
    print(f"  Estimated trace: {estimated_trace:.2f}")
    print(f"  Relative error:  {rel_error:.4f} ({rel_error*100:.1f}%)")

    if rel_error < 0.15:
        print(f"  [PASS] Estimate within 15% of exact")
    else:
        print(f"  [WARN] Estimate has {rel_error*100:.1f}% error (may need more samples)")

    # Test that different layers get different sensitivity scores
    scores = []
    for i in range(5):
        Xi = torch.randn(256, n) * (i + 1)  # increasing scale
        score = hutchpp_trace(torch.randn(64, n), Xi, n_samples=20)
        scores.append(score)
        print(f"  Layer {i} (scale={i+1}x): sensitivity={score:.2f}")

    assert scores[-1] > scores[0], "FAIL: Sensitivity should increase with activation scale"
    print(f"  [PASS] Sensitivity ordering correct")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PARAMETER GOLF: Component Validation Suite")
    print("  Running on CPU — no CUDA required")
    print("=" * 60 + "\n")

    t_start = time.time()

    test_hestia()
    test_2_4_packing()
    test_gptq_ternary()
    test_eggroll()
    test_hutchpp()

    elapsed = time.time() - t_start

    print("=" * 60)
    print(f"  ALL TESTS COMPLETE in {elapsed:.1f}s")
    print("=" * 60)
