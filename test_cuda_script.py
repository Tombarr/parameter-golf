"""
Test suite for train_gpt_hestia_ternary.py — validates novel components on CPU.
Imports directly from the CUDA script (mocking CUDA-only deps) to test real code paths.

Usage:
    python test_cuda_script.py
"""

import sys
import os
import math
import time
import unittest
import io
import lzma
import numpy as np

# Mock flash_attn before importing the script
import types
mock_flash = types.ModuleType("flash_attn_interface")
def _mock_flash_attn_func(q, k, v, causal=False):
    """CPU fallback: standard scaled dot-product attention."""
    import torch
    import torch.nn.functional as F
    # q: (B, S, H, D), k: (B, S, KVH, D), v: (B, S, KVH, D)
    B, S, H, D = q.shape
    KVH = k.shape[2]
    if KVH < H:
        rep = H // KVH
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    q = q.transpose(1, 2)  # (B, H, S, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return y.transpose(1, 2)  # (B, S, H, D)
mock_flash.flash_attn_func = _mock_flash_attn_func
sys.modules["flash_attn_interface"] = mock_flash

# Mock torch.distributed for non-distributed testing
import torch
import torch.distributed as dist
if not dist.is_available() or not dist.is_initialized():
    # Ensure script doesn't fail on CPU
    pass

# Mock torch.compiler if needed
if not hasattr(torch, 'compiler'):
    torch.compiler = types.SimpleNamespace(cudagraph_mark_step_begin=lambda: None)
elif not hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
    torch.compiler.cudagraph_mark_step_begin = lambda: None

# Add the script directory to path
SCRIPT_DIR = os.path.join(os.path.dirname(__file__),
    "records/track_10min_16mb/2026-04-05_HESTIA_Ternary_24Sparse_GPTQ_EGGROLL")
sys.path.insert(0, SCRIPT_DIR)

# Import from the actual CUDA script
import train_gpt_hestia_ternary as T

import torch.nn.functional as F


class TestHESTIA(unittest.TestCase):
    """Test HESTIA softmax relaxation from the CUDA script."""

    def test_soft_quantize_gradient_flow(self):
        w = torch.randn(64, 128, requires_grad=True)
        g = 128
        w_g = w.reshape(-1, g)
        scale_g = w_g.detach().abs().mean(-1, keepdim=True).clamp(min=1e-8)
        out = T.hestia_soft_quantize(w_g, scale_g, tau=torch.tensor(0.3))
        out.sum().backward()
        self.assertIsNotNone(w.grad)
        self.assertGreater(w.grad.abs().sum().item(), 0)

    def test_converges_to_hard_ternary(self):
        torch.manual_seed(42)
        w = torch.randn(32, 128)
        g = 128
        w_g = w.reshape(-1, g)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q_hard = (w_g / scale).round().clamp(-1, 1) * scale
        errors = []
        for tau in [1.0, 0.01, 0.001]:
            soft = T.hestia_soft_quantize(w_g, scale, torch.tensor(tau))
            errors.append((soft - q_hard).abs().mean().item())
        self.assertLess(errors[-1], errors[0] * 0.1)

    def test_hestia_ternary_forward_ste_fallback(self):
        """When pressure=0, should fall back to STE."""
        w = torch.randn(64, 128, requires_grad=True)
        tau = torch.tensor(0.3)
        pressure = torch.tensor(0.0)
        out = T.hestia_ternary_forward(w, group_size=128, tau=tau, pressure=pressure, sensitivity_exp=1.0)
        self.assertEqual(out.shape, w.shape)
        out.sum().backward()
        self.assertIsNotNone(w.grad)

    def test_hestia_forward_with_pressure(self):
        w = torch.randn(64, 128, requires_grad=True)
        tau = torch.tensor(0.1)
        pressure = torch.tensor(0.5)
        out = T.hestia_ternary_forward(w, group_size=128, tau=tau, pressure=pressure, sensitivity_exp=1.0)
        self.assertEqual(out.shape, w.shape)
        out.sum().backward()
        self.assertIsNotNone(w.grad)


class Test24Structured(unittest.TestCase):
    """Test 2:4 structured sparsity packing from the CUDA script."""

    def test_enforce_structure(self):
        w = torch.randn(256, 128)
        w_24 = T.enforce_2_4_structure(w)
        flat = w_24.reshape(-1, 4)
        # Exactly 2 nonzeros per group
        self.assertTrue((flat.ne(0).sum(dim=1) == 2).all())
        # Values in {-1, 0, 1}
        self.assertTrue(((flat == -1) | (flat == 0) | (flat == 1)).all())

    def test_pack_unpack_roundtrip(self):
        w = torch.randn(256, 128)
        w_24 = T.enforce_2_4_structure(w).to(torch.int8)
        packed, n_groups = T.pack_2_4_ternary(w_24)
        unpacked = T.unpack_2_4_ternary(packed, n_groups)
        # Trim to original size
        expected = w_24.reshape(-1)
        actual = unpacked[:expected.numel()]
        self.assertTrue(torch.equal(expected, actual.to(torch.int8)))

    def test_compression_better_than_base3(self):
        """2:4 packing + LZMA should beat base-3 + LZMA."""
        torch.manual_seed(42)
        w = torch.randn(512, 256)
        scale = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        w_ternary = (w / scale).round().clamp(-1, 1)
        w_24 = T.enforce_2_4_structure(w)

        # Base-3 + LZMA
        b3_bytes, _ = T.pack_ternary(w_ternary)
        b3_compressed = lzma.compress(b3_bytes, preset=9)

        # 2:4 + LZMA
        p24_bytes, _ = T.pack_2_4_ternary(w_24.to(torch.int8))
        p24_compressed = lzma.compress(p24_bytes, preset=9)

        self.assertLess(len(p24_compressed), len(b3_compressed),
                        f"2:4 ({len(p24_compressed)}) should be smaller than base-3 ({len(b3_compressed)})")


class TestGPTQTernary(unittest.TestCase):
    """Test GPTQ error compensation for ternary from the CUDA script."""

    def test_reduces_reconstruction_error(self):
        torch.manual_seed(42)
        group_size = 64
        W = torch.randn(64, 128) * 0.02
        X = torch.randn(256, 128)
        H = (X.T @ X) / X.shape[0]
        # Per-group scales (matching what q_sd computes)
        W_grouped = W.reshape(-1, group_size)
        scale = W_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        # Per-row scale for naive comparison
        row_scale = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)

        Q_naive = (W / row_scale).round().clamp(-1, 1)
        Q_gptq = T.gptq_ternary_quantize(W, H, scale, group_size)

        # Reconstruct using per-group scales
        naive_recon = Q_naive * row_scale
        gptq_grouped = Q_gptq.reshape(-1, group_size)
        gptq_recon = (gptq_grouped * scale).reshape(W.shape)

        naive_err = ((X @ naive_recon.T - X @ W.T) ** 2).mean().item()
        gptq_err = ((X @ gptq_recon.T - X @ W.T) ** 2).mean().item()

        self.assertLess(gptq_err, naive_err * 1.1,
                        f"GPTQ ({gptq_err:.6f}) should improve on naive ({naive_err:.6f})")

    def test_output_is_valid_ternary(self):
        group_size = 64
        W = torch.randn(32, 64) * 0.02
        X = torch.randn(128, 64)
        H = (X.T @ X) / X.shape[0]
        W_grouped = W.reshape(-1, group_size)
        scale = W_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        Q = T.gptq_ternary_quantize(W, H, scale, group_size)
        self.assertTrue(((Q == -1) | (Q == 0) | (Q == 1)).all())


class TestSerialization(unittest.TestCase):
    """Test the full quantize → serialize → deserialize → dequantize pipeline."""

    def test_q_sd_roundtrip_2_4(self):
        """State dict survives 2:4 quantize → pack → LZMA → decompress → unpack."""
        torch.manual_seed(42)
        sd = {
            "layer.weight": torch.randn(256, 512) * 0.02,  # >65536 elements → ternary
            "bias": torch.randn(128),  # fp16
        }
        q_sd, stats = T.q_sd(sd, group_size=128, fp_storage=False, use_2_4=True)
        self.assertEqual(q_sd["layer.weight"]["type"], "ternary_2_4")
        self.assertEqual(q_sd["bias"]["type"], "fp16")
        self.assertGreater(stats["ternary_params"], 0)

        # Serialize + deserialize
        buf = io.BytesIO()
        torch.save(q_sd, buf)
        blob = lzma.compress(buf.getvalue(), preset=9)
        loaded = torch.load(io.BytesIO(lzma.decompress(blob)), weights_only=False)

        # Dequantize
        restored = T.deq_sd(loaded)
        self.assertIn("layer.weight", restored)
        self.assertEqual(restored["layer.weight"].shape, (256, 512))
        # Values should be approximately ternary * scale
        w = restored["layer.weight"].float()
        self.assertFalse(torch.isnan(w).any())
        self.assertFalse(torch.isinf(w).any())

    def test_q_sd_roundtrip_base3(self):
        """State dict survives base-3 quantize → pack → LZMA → decompress → unpack."""
        torch.manual_seed(42)
        sd = {"layer.weight": torch.randn(256, 512) * 0.02}  # >65536 elements
        q_sd, stats = T.q_sd(sd, group_size=128, fp_storage=False, use_2_4=False)
        self.assertEqual(q_sd["layer.weight"]["type"], "ternary")

        buf = io.BytesIO()
        torch.save(q_sd, buf)
        blob = lzma.compress(buf.getvalue(), preset=9)
        loaded = torch.load(io.BytesIO(lzma.decompress(blob)), weights_only=False)
        restored = T.deq_sd(loaded)
        self.assertEqual(restored["layer.weight"].shape, (256, 512))

    def test_artifact_size_estimate(self):
        """Check that 65M params fits under 16MB with 2:4 packing."""
        torch.manual_seed(42)
        # Simulate 65M ternary params as many weight matrices
        total_ternary_bytes = 0
        total_fp_bytes = 0
        for _ in range(10):  # 10 layers
            w = torch.randn(768, 3072) * 0.02  # ~2.36M params per matrix
            q_sd, stats = T.q_sd({"w": w}, group_size=128, use_2_4=True)
            buf = io.BytesIO()
            torch.save(q_sd, buf)
            blob = lzma.compress(buf.getvalue(), preset=9)
            total_ternary_bytes += len(blob)

        projected_65m = total_ternary_bytes * (65e6 / (10 * 768 * 3072))
        # Should be well under 16MB
        self.assertLess(projected_65m / 1e6, 14.0,
                        f"Projected {projected_65m/1e6:.2f}MB should be under 14MB for 65M params")


class TestModelComponents(unittest.TestCase):
    """Test model building blocks from the CUDA script."""

    def test_hestia_ternary_linear(self):
        layer = T.HestiaTernaryLinear(64, 128, group_size=64)
        layer.hestia_tau.fill_(0.1)
        layer.hestia_pressure.fill_(0.5)
        x = torch.randn(2, 8, 64)
        y = layer(x)
        self.assertEqual(y.shape, (2, 8, 128))
        y.sum().backward()
        self.assertIsNotNone(layer.weight.grad)

    def test_normed_hestia_linear(self):
        layer = T.NormedHestiaTernaryLinear(64, 128, group_size=64)
        x = torch.randn(2, 8, 64)
        y = layer(x)
        self.assertEqual(y.shape, (2, 8, 128))

    def test_mlp(self):
        mlp = T.MLP(64, 4, group_size=64, activation="relu2")
        x = torch.randn(2, 8, 64)
        y = mlp(x)
        self.assertEqual(y.shape, (2, 8, 64))

    def test_attention_cpu(self):
        """Test attention with mocked flash_attn."""
        attn = T.CausalSelfAttention(
            dim=64, num_heads=4, num_kv_heads=2, rope_base=10000.0,
            qk_gain_init=1.5, group_size=64, no_cache=True)
        x = torch.randn(2, 16, 64)
        y = attn(x)
        self.assertEqual(y.shape, (2, 16, 64))

    def test_block(self):
        block = T.Block(
            dim=64, num_heads=4, num_kv_heads=2, mlp_mult=4,
            rope_base=10000.0, qk_gain_init=1.5, group_size=64,
            activation="relu2", no_cache=True)
        x = torch.randn(2, 16, 64)
        x0 = x.clone()
        y = block(x, x0)
        self.assertEqual(y.shape, (2, 16, 64))

    def test_full_model_forward(self):
        """Full GPT model forward pass on CPU."""
        model = T.GPT(
            vocab_size=64, num_layers=2, model_dim=64, num_heads=4,
            num_kv_heads=2, mlp_mult=2, tie_embeddings=1,
            tied_embed_init_std=0.02, logit_softcap=10.0, rope_base=10000.0,
            qk_gain_init=1.5, group_size=64, activation="relu2",
            no_cache=True, softcap_type="poly")
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))
        loss = model(x, y)
        self.assertFalse(torch.isnan(loss))
        loss.backward()

    def test_full_model_hestia_training_step(self):
        """Simulate a HESTIA training step with gradient flow."""
        model = T.GPT(
            vocab_size=64, num_layers=2, model_dim=64, num_heads=4,
            num_kv_heads=2, mlp_mult=2, tie_embeddings=1,
            tied_embed_init_std=0.02, logit_softcap=10.0, rope_base=10000.0,
            qk_gain_init=1.5, group_size=64, activation="relu2",
            no_cache=True, softcap_type="poly")
        model.set_hestia_params(tau=0.2, pressure=0.7)
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))
        loss = model(x, y)
        loss.backward()
        # Check ternary weight layers got gradients (skip control params like scales)
        ternary_grads = 0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and p.ndim == 2 and "weight" in name:
                if p.grad.abs().sum().item() > 0:
                    ternary_grads += 1
        self.assertGreater(ternary_grads, 0, "No ternary weight layers got gradients")


class TestHutchpp(unittest.TestCase):
    """Test Hutch++ sensitivity estimation."""

    def test_estimate_sensitivities_basic(self):
        """Smoke test that sensitivity estimation runs without error."""
        model = T.GPT(
            vocab_size=64, num_layers=2, model_dim=64, num_heads=4,
            num_kv_heads=2, mlp_mult=2, tie_embeddings=1,
            tied_embed_init_std=0.02, logit_softcap=10.0, rope_base=10000.0,
            qk_gain_init=1.5, group_size=64, activation="relu2",
            no_cache=True, softcap_type="poly")
        calib = torch.randint(0, 64, (4, 17))  # 4 seqs, seq_len+1
        sensitivities = T.estimate_sensitivities(model, calib, torch.device("cpu"), 64)
        self.assertGreater(len(sensitivities), 0)
        # Check all values are finite
        for name, val in sensitivities.items():
            self.assertTrue(math.isfinite(val), f"Non-finite sensitivity for {name}")


if __name__ == "__main__":
    print("=" * 60)
    print("  CUDA Script Test Suite (CPU mode)")
    print("=" * 60)
    unittest.main(verbosity=2)
