"""
Regression tests for bugs discovered during RunPod validation.
Each test targets a specific failure mode we hit on real hardware.

Usage:
    python test_regressions.py
"""

import sys
import os
import io
import lzma
import math
import types
import unittest

import torch
import torch.nn.functional as F

# ── Mock CUDA-only deps ──────────────────────────────────────────
mock_flash = types.ModuleType("flash_attn_interface")
def _mock_fa(q, k, v, causal=False):
    B, S, H, D = q.shape
    KVH = k.shape[2]
    if KVH < H:
        k = k.repeat_interleave(H // KVH, dim=2)
        v = v.repeat_interleave(H // KVH, dim=2)
    return F.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=causal
    ).transpose(1,2)
mock_flash.flash_attn_func = _mock_fa
sys.modules["flash_attn_interface"] = mock_flash
if not hasattr(torch, 'compiler'):
    torch.compiler = types.SimpleNamespace(cudagraph_mark_step_begin=lambda: None)
elif not hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
    torch.compiler.cudagraph_mark_step_begin = lambda: None

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "records/track_10min_16mb/2026-04-05_HESTIA_Ternary_24Sparse_GPTQ_EGGROLL")
sys.path.insert(0, SCRIPT_DIR)
import train_gpt_hestia_ternary as T


class TestBase3DequantNoCorruption(unittest.TestCase):
    """
    BUG: base3 deq_sd used `scale / q_absmean` which corrupts GPTQ-optimized weights.
    The q_absmean correction changes the effective magnitude of weights that GPTQ
    carefully tuned, causing roundtrip loss to explode (4.4 → 12.1).
    FIX: use `q * scale` directly.
    """

    def test_roundtrip_preserves_weight_magnitude(self):
        """After quantize→serialize→deserialize, weight magnitudes should be close to original."""
        torch.manual_seed(42)
        W = torch.randn(256, 512) * 0.02  # >65536 elements
        sd = {"layer.weight": W}

        q_obj, _ = T.q_sd(sd, group_size=128, use_2_4=False)
        buf = io.BytesIO()
        torch.save(q_obj, buf)
        blob = lzma.compress(buf.getvalue(), preset=9)
        loaded = torch.load(io.BytesIO(lzma.decompress(blob)), weights_only=False)
        restored = T.deq_sd(loaded)

        W_restored = restored["layer.weight"].float()
        # Ternary quantization will have error, but magnitude should be in the right ballpark
        orig_mag = W.abs().mean().item()
        restored_mag = W_restored.abs().mean().item()
        ratio = restored_mag / orig_mag
        self.assertGreater(ratio, 0.5, f"Restored magnitude {restored_mag:.4f} too small vs original {orig_mag:.4f}")
        self.assertLess(ratio, 2.0, f"Restored magnitude {restored_mag:.4f} too large vs original {orig_mag:.4f}")

    def test_roundtrip_loss_reasonable(self):
        """Full model roundtrip should not catastrophically degrade loss."""
        torch.manual_seed(42)
        model = T.GPT(
            vocab_size=256, num_layers=2, model_dim=256, num_heads=4,
            num_kv_heads=2, mlp_mult=2, tie_embeddings=1, tied_embed_init_std=0.02,
            logit_softcap=10.0, rope_base=10000.0, qk_gain_init=1.5,
            group_size=128, activation="relu2", no_cache=True, softcap_type="poly")

        x = torch.randint(0, 256, (4, 32))
        y = torch.randint(0, 256, (4, 32))
        model.eval()
        with torch.no_grad():
            pre_loss = model(x, y).item()

        sd = model.state_dict()
        sd.pop("lm_head.weight", None)
        q_obj, _ = T.q_sd(sd, group_size=128, use_2_4=False)
        buf = io.BytesIO()
        torch.save(q_obj, buf)
        blob = lzma.compress(buf.getvalue(), preset=9)
        loaded = torch.load(io.BytesIO(lzma.decompress(blob)), weights_only=False)
        restored_sd = T.deq_sd(loaded)
        restored_sd["lm_head.weight"] = restored_sd["tok_emb.weight"]

        model2 = T.GPT(
            vocab_size=256, num_layers=2, model_dim=256, num_heads=4,
            num_kv_heads=2, mlp_mult=2, tie_embeddings=1, tied_embed_init_std=0.02,
            logit_softcap=10.0, rope_base=10000.0, qk_gain_init=1.5,
            group_size=128, activation="relu2", no_cache=True, softcap_type="poly")
        model2.load_state_dict(restored_sd, strict=False)
        model2.eval()
        with torch.no_grad():
            post_loss = model2(x, y).item()

        gap = post_loss - pre_loss
        self.assertLess(gap, 2.0,
            f"Roundtrip gap {gap:.2f} is catastrophic (pre={pre_loss:.2f}, post={post_loss:.2f}). "
            f"Likely q_absmean corruption bug.")


class Test24MaskUsesContinuousWeights(unittest.TestCase):
    """
    BUG: 2:4 enforcement applied topk on ternary values {-1,0,1} where all nonzeros
    have magnitude 1, so topk picked arbitrarily — randomly zeroing half the weights.
    FIX: use original continuous weights to decide which 2 of 4 to keep.
    """

    def test_mask_preserves_largest_magnitude_weights(self):
        """The 2 kept weights per group of 4 should be the ones with largest original magnitude."""
        torch.manual_seed(42)
        # Create weights with clear magnitude differences, large enough for ternary threshold
        W_pattern = torch.tensor([[10.0, 0.1, 8.0, 0.2],
                                  [0.3, 9.0, 0.1, 7.0]], dtype=torch.float32)
        W = W_pattern.repeat(128, 128)  # 256x512 = 131072 > 65536
        sd = {"layer.weight": W}

        q_obj, _ = T.q_sd(sd, group_size=128, use_2_4=True)
        entry = q_obj["layer.weight"]
        self.assertEqual(entry["type"], "ternary_2_4")

        # Unpack and check first row
        q = T.unpack_2_4_ternary(entry["packed"], entry["n_groups"])
        first_group = q[:4].tolist()
        # Positions 0 (mag=10) and 2 (mag=8) should be nonzero
        self.assertNotEqual(first_group[0], 0, "Position 0 (mag=10) should be kept")
        self.assertNotEqual(first_group[2], 0, "Position 2 (mag=8) should be kept")
        self.assertEqual(first_group[1], 0, "Position 1 (mag=0.1) should be zeroed")
        self.assertEqual(first_group[3], 0, "Position 3 (mag=0.2) should be zeroed")

    def test_2_4_roundtrip_loss_not_catastrophic(self):
        """2:4 quantized model roundtrip should not destroy loss."""
        torch.manual_seed(42)
        W = torch.randn(256, 512) * 0.02
        sd = {"layer.weight": W}

        q_obj, _ = T.q_sd(sd, group_size=128, use_2_4=True)
        buf = io.BytesIO()
        torch.save(q_obj, buf)
        blob = lzma.compress(buf.getvalue(), preset=9)
        loaded = torch.load(io.BytesIO(lzma.decompress(blob)), weights_only=False)
        restored = T.deq_sd(loaded)

        W_restored = restored["layer.weight"].float()
        # MSE should be reasonable — not zero (we lose info) but not huge
        mse = ((W - W_restored) ** 2).mean().item()
        orig_var = W.var().item()
        # MSE should be less than original variance (we're reconstructing, not random)
        self.assertLess(mse, orig_var,
            f"MSE {mse:.6f} >= variance {orig_var:.6f} — reconstruction is noise-level")


class TestTokenDtypeCompat(unittest.TestCase):
    """
    BUG: val_tokens loaded as uint16, but embedding/indexing ops need int64.
    Affected: EGGROLL eval batch, sliding eval fancy indexing.
    FIX: cast to long before use.
    """

    def test_eggroll_handles_uint16_tokens(self):
        """EGGROLL should work with uint16 val_tokens (as loaded from shards)."""
        torch.manual_seed(42)
        model = T.GPT(
            vocab_size=256, num_layers=2, model_dim=64, num_heads=4,
            num_kv_heads=2, mlp_mult=2, tie_embeddings=1, tied_embed_init_std=0.02,
            logit_softcap=10.0, rope_base=10000.0, qk_gain_init=1.5,
            group_size=64, activation="relu2", no_cache=True, softcap_type="poly")

        # Simulate shard-loaded uint16 tokens
        val_tokens = torch.randint(0, 256, (2048,), dtype=torch.uint16)
        # This should not crash
        try:
            # Reproduce the EGGROLL eval batch creation
            seq_len = 32
            n_seqs = min(4, (val_tokens.numel() - 1) // seq_len)
            eval_x = val_tokens[:n_seqs * seq_len].reshape(n_seqs, seq_len).to(dtype=torch.long)
            eval_y = val_tokens[1:n_seqs * seq_len + 1].reshape(n_seqs, seq_len).to(dtype=torch.long)
            model.eval()
            with torch.no_grad():
                loss = model(eval_x, eval_y)
            self.assertFalse(torch.isnan(loss), "Loss is NaN")
        except RuntimeError as e:
            self.fail(f"EGGROLL failed with uint16 tokens: {e}")

    def test_sliding_eval_handles_uint16_tokens(self):
        """Sliding eval fancy indexing should work with uint16 val_tokens."""
        val_tokens = torch.randint(0, 256, (2048,), dtype=torch.uint16)
        seq_len = 32
        stride = 8
        batch_size = 4
        all_starts = list(range(0, val_tokens.numel() - seq_len - 1, stride))[:batch_size]
        starts_t = torch.tensor(all_starts, dtype=torch.int64)
        offsets = torch.arange(seq_len + 1, dtype=torch.int64)
        indices = starts_t.unsqueeze(1) + offsets.unsqueeze(0)

        try:
            # This was the exact line that crashed: val_tokens[indices]
            local_batch = val_tokens.long()[indices]
            self.assertEqual(local_batch.shape, (batch_size, seq_len + 1))
        except (NotImplementedError, RuntimeError) as e:
            self.fail(f"Sliding eval indexing failed with uint16: {e}")


class TestHutchCalibBatchSizing(unittest.TestCase):
    """
    BUG: Hutch++ calib batch needed 8*(seq_len+1) tokens but only allocated 8*seq_len+1.
    shape '[8, -1]' invalid for input of size 8193 (needed 8200).
    FIX: allocate n_calib_seqs * (seq_len + 1) tokens.
    """

    def test_calib_batch_shape_correct(self):
        """Verify the calib batch reshape works for various seq_lens."""
        for seq_len in [32, 64, 128, 256, 512, 1024]:
            n_seqs = 8
            tokens_needed = n_seqs * (seq_len + 1)
            tokens = torch.randint(0, 1024, (tokens_needed,))
            # This is the fixed reshape — should not crash
            try:
                batch = tokens[:tokens_needed].reshape(n_seqs, -1)
                self.assertEqual(batch.shape, (n_seqs, seq_len + 1),
                    f"Wrong shape for seq_len={seq_len}: {batch.shape}")
            except RuntimeError as e:
                self.fail(f"Reshape failed for seq_len={seq_len}: {e}")

    def test_old_formula_would_fail(self):
        """Verify the OLD formula (seq_len*8+1) doesn't work."""
        seq_len = 1024
        old_size = seq_len * 8 + 1  # = 8193
        with self.assertRaises(RuntimeError):
            torch.zeros(old_size).reshape(8, -1)  # 8193 not divisible by 8


class TestGenerateAutocast(unittest.TestCase):
    """
    BUG: generate() passed float32 tensors to flash_attn which requires bf16/fp16.
    FIX: wrap forward pass in autocast.
    """

    def test_generate_produces_valid_tokens(self):
        """Generate should produce tokens in [0, vocab_size) without dtype errors."""
        torch.manual_seed(42)
        model = T.GPT(
            vocab_size=64, num_layers=2, model_dim=64, num_heads=4,
            num_kv_heads=2, mlp_mult=2, tie_embeddings=1, tied_embed_init_std=0.02,
            logit_softcap=10.0, rope_base=10000.0, qk_gain_init=1.5,
            group_size=64, activation="relu2", no_cache=True, softcap_type="poly")

        tokens = model.generate(n_seqs=2, max_len=16, temperature=0.8,
                                device=torch.device("cpu"))
        self.assertEqual(tokens.shape[0], 2)
        self.assertEqual(tokens.shape[1], 16)
        self.assertTrue((tokens >= 0).all())
        self.assertTrue((tokens < 64).all())

    def test_generate_dtype_is_long(self):
        """Generated tokens should be int64 (long), not float."""
        torch.manual_seed(42)
        model = T.GPT(
            vocab_size=64, num_layers=2, model_dim=64, num_heads=4,
            num_kv_heads=2, mlp_mult=2, tie_embeddings=1, tied_embed_init_std=0.02,
            logit_softcap=10.0, rope_base=10000.0, qk_gain_init=1.5,
            group_size=64, activation="relu2", no_cache=True, softcap_type="poly")

        tokens = model.generate(n_seqs=1, max_len=8, device=torch.device("cpu"))
        self.assertEqual(tokens.dtype, torch.long)


class TestHESTIACompileCompat(unittest.TestCase):
    """
    BUG: HESTIA used Python float globals (_HESTIA_TAU) that caused torch.compile
    to recompile every step via .item() guard breaks.
    FIX: tensor buffers on HestiaTernaryLinear, torch.where instead of Python if/else.
    """

    def test_hestia_params_are_tensor_buffers(self):
        """HestiaTernaryLinear should have tensor buffers, not rely on globals."""
        layer = T.HestiaTernaryLinear(64, 128, group_size=64)
        self.assertTrue(hasattr(layer, 'hestia_tau'))
        self.assertTrue(hasattr(layer, 'hestia_pressure'))
        self.assertIsInstance(layer.hestia_tau, torch.Tensor)
        self.assertIsInstance(layer.hestia_pressure, torch.Tensor)

    def test_set_hestia_params_updates_all_layers(self):
        """GPT.set_hestia_params should update buffers on every ternary layer."""
        model = T.GPT(
            vocab_size=64, num_layers=2, model_dim=64, num_heads=4,
            num_kv_heads=2, mlp_mult=2, tie_embeddings=1, tied_embed_init_std=0.02,
            logit_softcap=10.0, rope_base=10000.0, qk_gain_init=1.5,
            group_size=64, activation="relu2", no_cache=True, softcap_type="poly")

        model.set_hestia_params(tau=0.15, pressure=0.8)
        for m in model.modules():
            if isinstance(m, T.HestiaTernaryLinear):
                self.assertAlmostEqual(m.hestia_tau.item(), 0.15, places=5)
                self.assertAlmostEqual(m.hestia_pressure.item(), 0.8, places=5)

    def test_hestia_forward_no_item_calls(self):
        """hestia_ternary_forward should use torch.where, not .item() branching."""
        import inspect
        source = inspect.getsource(T.hestia_ternary_forward)
        self.assertNotIn('.item()', source,
            "hestia_ternary_forward still uses .item() — will cause torch.compile graph breaks")
        self.assertIn('torch.where', source,
            "hestia_ternary_forward should use torch.where for compile compatibility")


class TestFA3FallbackDetection(unittest.TestCase):
    """
    BUG: FA3 was installed on RTX 3090 but crashes at runtime on non-Hopper GPUs.
    Import-only fallback didn't catch this.
    FIX: check compute capability >= 9.0 before using FA3.
    """

    def test_sdpa_fallback_produces_correct_shape(self):
        """SDPA fallback should produce same output shape as FA3."""
        q = torch.randn(2, 16, 4, 32)  # (B, S, H, D)
        k = torch.randn(2, 16, 2, 32)  # GQA: fewer KV heads
        v = torch.randn(2, 16, 2, 32)

        out = T._sdpa_fallback(q, k, v, causal=True)
        self.assertEqual(out.shape, (2, 16, 4, 32))

    def test_sdpa_fallback_is_causal(self):
        """SDPA fallback with causal=True should produce different output than non-causal."""
        torch.manual_seed(42)
        q = torch.randn(1, 8, 1, 16)
        k = torch.randn(1, 8, 1, 16)
        v = torch.randn(1, 8, 1, 16)

        out_causal = T._sdpa_fallback(q, k, v, causal=True)
        out_full = T._sdpa_fallback(q, k, v, causal=False)
        # Overall outputs should differ (causal masking changes attention weights)
        self.assertFalse(torch.allclose(out_causal, out_full, atol=1e-5),
            "Causal and full attention should produce different outputs")


if __name__ == "__main__":
    print("=" * 60)
    print("  REGRESSION TESTS — RunPod-Discovered Bugs")
    print("=" * 60)
    unittest.main(verbosity=2)
