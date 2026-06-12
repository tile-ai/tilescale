"""Numerical reference + preprocessing for the TileScale FP8/FP4 Mega MoE port.

Everything here is pure PyTorch and mirrors DeepGEMM's host-side math so that the
TileScale kernels can be verified against an independent ground truth.

The pieces that are aligned with DeepGEMM (``deep_gemm/utils/math.py`` and
``deep_gemm/mega/__init__.py``):

* ``ceil_to_ue8m0`` / ``pack_ue8m0_to_int``  - UE8M0 (power-of-two) scale factors.
* ``per_token_cast_to_fp8`` (e4m3, ``448`` clamp) and ``per_token_cast_to_fp4``
  (e2m1, ``6`` clamp) with ``gran_k`` granularity along K.
* ``transform_weights_for_mega_moe`` - the gate/up interleave for the L1 weights.
* ``swiglu`` epilogue (silu(gate) * up * topk_weight with optional clamp) which
  matches the fused kernel's L1 activation.
* ``mega_moe_reference`` - the full dispatch -> L1 -> SwiGLU -> L2 -> combine
  numerics for a single rank (multi-rank reduces to the same math because
  dispatch/combine are pure gather/scatter of token rows).
"""

from __future__ import annotations

from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Small integer helpers
# ---------------------------------------------------------------------------
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


# ---------------------------------------------------------------------------
# UE8M0 scale-factor helpers (aligned with deep_gemm/utils/math.py)
# ---------------------------------------------------------------------------
def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round a positive FP32 scale up to the next power of two (UE8M0)."""
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def pack_ue8m0_to_int(sf_fp32: torch.Tensor) -> torch.Tensor:
    """Pack 4 consecutive UE8M0 exponent bytes (along K) into one int32.

    ``sf_fp32`` has shape ``[mn, num_k_groups]`` where ``num_k_groups`` is a
    multiple of 4. Output is ``[mn, num_k_groups // 4]`` int32.
    """
    assert sf_fp32.dtype == torch.float32 and sf_fp32.size(-1) % 4 == 0
    sf_u8 = (sf_fp32.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    words = sf_u8.to(torch.int64)
    packed = (words[..., 0::4] | (words[..., 1::4] << 8) | (words[..., 2::4] << 16) | (words[..., 3::4] << 24))
    return packed.to(torch.int32).contiguous()


def unpack_int_to_ue8m0(packed: torch.Tensor, num_k_blocks: int) -> torch.Tensor:
    """Inverse of :func:`pack_ue8m0_to_int`, returns UE8M0 exponent bytes."""
    words = packed.contiguous().to(torch.int64)
    out_shape = (*packed.shape[:-1], packed.shape[-1] * 4)
    unpacked = torch.empty(out_shape, device=packed.device, dtype=torch.uint8)
    for i in range(4):
        unpacked[..., i::4] = ((words >> (8 * i)) & 0xFF).to(torch.uint8)
    return unpacked[..., :num_k_blocks].contiguous()


def ue8m0_to_scale(exp_bytes: torch.Tensor) -> torch.Tensor:
    """Convert UE8M0 exponent bytes to FP32 multiplicative scales (2 ** (e - 127))."""
    return torch.pow(2.0, exp_bytes.to(torch.float32) - 127.0)


# ---------------------------------------------------------------------------
# FP8 / FP4 quantization (aligned with deep_gemm/utils/math.py)
# ---------------------------------------------------------------------------
def per_token_cast_to_fp8(
    x: torch.Tensor,
    use_ue8m0: bool = True,
    gran_k: int = 32,
    use_packed_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast ``[M, N]`` BF16/FP32 to e4m3 with per-``gran_k`` scale factors."""
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
    return x_fp8, (pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf)


# FP4 E2M1 representable magnitudes and their decision boundaries.
_FP4_E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Quantize FP32 to packed-nibble E2M1 codes (sign bit + 3-bit magnitude)."""
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor(_FP4_E2M1_BOUNDARIES, device=x.device, dtype=torch.float32)
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def per_token_cast_to_fp4(
    x: torch.Tensor,
    use_ue8m0: bool = True,
    gran_k: int = 32,
    use_packed_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast ``[M, N]`` to packed E2M1 (``int8 [M, N // 2]``) + per-``gran_k`` SF."""
    m, n = x.shape
    assert n % 2 == 0
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 6.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, : n // 2].contiguous().to(torch.int8), (pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf)


def fp4_packed_to_float(packed: torch.Tensor, logical_k: int) -> torch.Tensor:
    """Dequantize packed E2M1 ``int8 [M, K // 2]`` back to FP32 ``[M, K]``."""
    u = packed.contiguous().view(torch.uint8)
    lut = torch.tensor(_FP4_E2M1_VALUES, device=u.device, dtype=torch.float32)
    lut = torch.cat([lut, -lut])  # index 8..15 are negatives
    lo = lut[(u & 0x0F).long()]
    hi = lut[((u >> 4) & 0x0F).long()]
    out = torch.empty((u.shape[0], logical_k), device=u.device, dtype=torch.float32)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


# ---------------------------------------------------------------------------
# Weight transform (aligned with deep_gemm/mega/__init__.py)
# ---------------------------------------------------------------------------
def interleave_gate_up(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    """Interleave the gate/up halves of an L1 weight/SF tensor along N.

    ``[gate | up]`` (N split in half) becomes
    ``[gate[0:gran], up[0:gran], gate[gran:2gran], up[gran:2gran], ...]``.
    """
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))


# ---------------------------------------------------------------------------
# SwiGLU activation (matches the fused kernel L1 epilogue)
# ---------------------------------------------------------------------------
def swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    weight: torch.Tensor,
    activation_clamp: float | None = None,
    via_bf16: bool = True,
) -> torch.Tensor:
    """silu(clamp_hi(gate)) * clamp(up) * weight, optionally rounding to bf16 first."""
    if via_bf16:
        gate = gate.to(torch.bfloat16)
        up = up.to(torch.bfloat16)
    if activation_clamp is not None:
        c = activation_clamp
        gate = torch.minimum(gate, torch.full_like(gate, c))
        up = torch.clamp(up, -c, c)
    gate = gate.float()
    up = up.float()
    silu = gate / (1.0 + torch.exp(-gate))
    return silu * up * weight.float().unsqueeze(-1)


# ---------------------------------------------------------------------------
# Full Mega MoE numerical reference (single rank semantics)
# ---------------------------------------------------------------------------
def mega_moe_reference(
    x_fp8: torch.Tensor,
    x_sf_packed: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_w_fp4: torch.Tensor,
    l1_w_sf_packed: torch.Tensor,
    l2_w_fp4: torch.Tensor,
    l2_w_sf_packed: torch.Tensor,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
    gran_k: int = 32,
    activation_clamp: float | None = None,
) -> torch.Tensor:
    """Compute final ``[num_tokens, hidden]`` BF16 output for one rank.

    Args mirror what the symmetric buffer holds. ``l*_w_fp4`` are
    ``[E, N, K // 2]`` packed E2M1 weights, ``l*_w_sf_packed`` are
    ``[E, N, K // (gran_k*4)]`` packed UE8M0 ints. ``topk_idx`` uses ``-1`` for
    masked selections (which contribute zero).
    """
    device = x_fp8.device
    num_tokens = x_fp8.size(0)
    e = l1_w_fp4.size(0)

    # Dequantize activations once (x is shared across all selected experts).
    x_sf_k = hidden // gran_k
    x_sf = ue8m0_to_scale(unpack_int_to_ue8m0(x_sf_packed, x_sf_k))  # [tokens, hidden/gran_k]
    x_f32 = x_fp8.float().view(num_tokens, hidden // gran_k, gran_k) * x_sf.unsqueeze(2)
    x_f32 = x_f32.view(num_tokens, hidden)

    def dequant_weight(w_fp4, w_sf_packed, n, k):
        w = fp4_packed_to_float(w_fp4.reshape(e * n, k // 2), k).view(e, n, k)
        sf = ue8m0_to_scale(unpack_int_to_ue8m0(w_sf_packed, k // gran_k))  # [e, n, k/gran]
        w = w.view(e, n, k // gran_k, gran_k) * sf.unsqueeze(-1)
        return w.view(e, n, k)

    l1_w = dequant_weight(l1_w_fp4, l1_w_sf_packed, intermediate_hidden * 2, hidden)
    l2_w = dequant_weight(l2_w_fp4, l2_w_sf_packed, hidden, intermediate_hidden)

    y = torch.zeros((num_tokens, hidden), device=device, dtype=torch.float32)
    num_topk = topk_idx.size(1)
    for t in range(num_tokens):
        for s in range(num_topk):
            expert = int(topk_idx[t, s].item())
            if expert < 0:
                continue
            w = float(topk_weights[t, s].item())
            # L1: [2I, H] @ [H] -> [2I], plain [gate | up] split (no interleave).
            l1_out = l1_w[expert] @ x_f32[t]
            half = intermediate_hidden
            g = l1_out[:half]
            u = l1_out[half:]
            act = swiglu(g.unsqueeze(0), u.unsqueeze(0), torch.tensor([w], device=device),
                         activation_clamp=activation_clamp).squeeze(0)  # [I]
            # Re-quantize to fp8 then dequant (mirrors the kernel's L2 input path).
            act_fp8, act_sf = per_token_cast_to_fp8(act.unsqueeze(0), gran_k=gran_k, use_packed_ue8m0=True)
            act_sf = ue8m0_to_scale(unpack_int_to_ue8m0(act_sf, intermediate_hidden // gran_k))
            act_f32 = (act_fp8.float().view(1, intermediate_hidden // gran_k, gran_k) * act_sf.unsqueeze(2)).view(intermediate_hidden)
            # L2: [H, I] @ [I] -> [H]
            l2_out = l2_w[expert] @ act_f32
            y[t] += l2_out
    return y.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Self-test: verify the reference is internally consistent.
# ---------------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)
    device = "cuda"
    num_tokens, hidden, intermediate_hidden = 8, 256, 128
    num_experts, num_topk = 4, 2
    gran_k = 32

    x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    l1_w = torch.randn(num_experts, intermediate_hidden * 2, hidden, device=device, dtype=torch.bfloat16) * 0.1
    l2_w = torch.randn(num_experts, hidden, intermediate_hidden, device=device, dtype=torch.bfloat16) * 0.1
    scores = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1)

    x_fp8, x_sf = per_token_cast_to_fp8(x, gran_k=gran_k, use_packed_ue8m0=True)
    l1_w_fp4, l1_w_sf = per_token_cast_to_fp4(l1_w.view(num_experts * intermediate_hidden * 2, hidden), gran_k=gran_k, use_packed_ue8m0=True)
    l1_w_fp4 = l1_w_fp4.view(num_experts, intermediate_hidden * 2, hidden // 2)
    l1_w_sf = l1_w_sf.view(num_experts, intermediate_hidden * 2, hidden // (gran_k * 4))
    l2_w_fp4, l2_w_sf = per_token_cast_to_fp4(l2_w.view(num_experts * hidden, intermediate_hidden), gran_k=gran_k, use_packed_ue8m0=True)
    l2_w_fp4 = l2_w_fp4.view(num_experts, hidden, intermediate_hidden // 2)
    l2_w_sf = l2_w_sf.view(num_experts, hidden, intermediate_hidden // (gran_k * 4))

    y = mega_moe_reference(
        x_fp8, x_sf, topk_idx, topk_weights,
        l1_w_fp4, l1_w_sf, l2_w_fp4, l2_w_sf,
        hidden, intermediate_hidden, num_experts, gran_k=gran_k,
    )

    # Independent high-precision reference (no quantization on the activation path).
    x_f32 = x.float()
    y_hp = torch.zeros(num_tokens, hidden, device=device)
    for t in range(num_tokens):
        for s in range(num_topk):
            ex = int(topk_idx[t, s].item())
            l1 = (l1_w[ex].float() @ x_f32[t])
            g, u = l1[:intermediate_hidden], l1[intermediate_hidden:]
            silu = g / (1.0 + torch.exp(-g))
            act = silu * u * float(topk_weights[t, s])
            y_hp[t] += l2_w[ex].float() @ act

    cos = torch.nn.functional.cosine_similarity(y.float().flatten(), y_hp.flatten(), dim=0)
    print(f"reference self-test: cosine(quantized, high-precision) = {cos.item():.4f}")
    assert cos.item() > 0.95, "reference numerics drifted too far from high-precision"
    print("reference self-test passed")


if __name__ == "__main__":
    _self_test()
