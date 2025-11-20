import torch

# Minimal FP8 KV-cache quantizer (copied from tests/quant.py)
def _quantize_k_cache_fp8(
    input_k_cache: torch.Tensor,  # (num_blocks, block_size, h_k, d)
    dv: int,
    tile_size: int = 128,
):
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, d = input_k_cache.shape
    assert h_k == 1
    x = input_k_cache.squeeze(2)  # [num_blocks, block_size, d]
    input_elem_size = x.element_size()

    result = torch.empty(
        (num_blocks, block_size, dv + num_tiles * 4 + input_elem_size * (d - dv)),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    result_k_nope_part = result[..., :dv]
    result_k_scale_factor = result[..., dv : dv + num_tiles * 4].view(torch.float32)
    result_k_rope_part = result[..., dv + num_tiles * 4 :].view(x.dtype)
    result_k_rope_part[:] = x[..., dv:]

    for tile_idx in range(0, num_tiles):
        cur_scale_inv = torch.abs(x[..., tile_idx * tile_size : (tile_idx + 1) * tile_size]).max(dim=-1).values / 448.0
        result_k_scale_factor[:, :, tile_idx] = cur_scale_inv
        cur_scale_inv = cur_scale_inv.unsqueeze(-1)
        cur_quant_nope = (x[..., tile_idx * tile_size : (tile_idx + 1) * tile_size].float() / cur_scale_inv.float()).to(
            torch.float8_e4m3fn
        )
        result_k_nope_part[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = cur_quant_nope

    result = result.view(num_blocks, block_size, 1, -1)
    return result