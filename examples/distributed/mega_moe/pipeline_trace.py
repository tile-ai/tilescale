"""Low-overhead GPU timestamp tracing for the SM90 Mega MoE example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


TRACE_FIELDS = 4
L1_TRACE_ROLES = 4
L2_TRACE_ROLES = 3

GLOBAL_TIMER_SOURCE = r"""
extern "C" __device__ __forceinline__ long long tl_globaltimer_ns() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return static_cast<long long>(value);
}
"""


@dataclass(frozen=True)
class TraceRange:
    cta: int
    track: str
    phase: str
    start_ns: int
    end_ns: int


PHASE_COLORS = {
    "metadata": "#64748b",
    "remote get": "#2563eb",
    "arrival wait": "#eab308",
    "stage wait": "#f97316",
    "TMA": "#0d9488",
    "GEMM": "#dc2626",
    "epilogue": "#9333ea",
    "scatter": "#0284c7",
    "reduce": "#16a34a",
}


def trace_schedule_steps(
    num_experts_per_rank: int,
    num_experts_per_wave: int,
    capacity: int,
    block_m: int,
    num_n_blocks: int,
    num_sms: int,
) -> int:
    num_waves = num_experts_per_rank // num_experts_per_wave
    max_wave_tiles = num_experts_per_wave * ((capacity + block_m - 1) // block_m) * num_n_blocks
    max_rounds = (max_wave_tiles + num_sms - 1) // num_sms
    return num_waves * max_rounds


def _append_range(
    ranges: list[TraceRange],
    values: Sequence[int],
    begin: int,
    end: int,
    cta: int,
    track: str,
    phase: str,
) -> None:
    start_ns = int(values[begin])
    end_ns = int(values[end])
    if start_ns > 0 and end_ns >= start_ns:
        ranges.append(TraceRange(cta, track, phase, start_ns, end_ns))


def _collect_l1(trace, selected_ctas: Sequence[int]) -> list[TraceRange]:
    values = trace.detach().cpu().tolist()
    ranges: list[TraceRange] = []
    for cta in selected_ctas:
        for dispatch_warp in range(2):
            fields = values[cta][dispatch_warp][0]
            track = f"dispatch{dispatch_warp}"
            _append_range(ranges, fields, 0, 1, cta, track, "metadata")
            _append_range(ranges, fields, 2, 3, cta, track, "remote get")
        for fields in values[cta][2]:
            _append_range(ranges, fields, 0, 1, cta, "producer", "arrival wait")
            _append_range(ranges, fields, 1, 2, cta, "producer", "TMA")
        for fields in values[cta][3]:
            _append_range(ranges, fields, 3, 0, cta, "math", "stage wait")
            _append_range(ranges, fields, 0, 1, cta, "math", "GEMM")
            _append_range(ranges, fields, 1, 2, cta, "math", "epilogue")
    return ranges


def _collect_l2(trace, selected_ctas: Sequence[int]) -> list[TraceRange]:
    values = trace.detach().cpu().tolist()
    ranges: list[TraceRange] = []
    for cta in selected_ctas:
        for fields in values[cta][0]:
            _append_range(ranges, fields, 0, 1, cta, "producer", "TMA")
        for fields in values[cta][1]:
            _append_range(ranges, fields, 3, 0, cta, "math", "stage wait")
            _append_range(ranges, fields, 0, 1, cta, "math", "GEMM")
            _append_range(ranges, fields, 1, 2, cta, "math", "scatter")
        fields = values[cta][2][0]
        _append_range(ranges, fields, 0, 1, cta, "reduce", "reduce")
    return ranges


def _draw_panel(draw, ranges: Sequence[TraceRange], selected_ctas: Sequence[int], tracks: Sequence[str], box) -> None:
    from PIL import ImageFont

    left, top, right, bottom = box
    font = ImageFont.load_default()
    label_width = 128
    axis_left = left + label_width
    axis_right = right - 12
    title_height = 42
    row_height = max(16, (bottom - top - title_height - 24) // max(1, len(selected_ctas) * len(tracks)))
    timestamps = [value for item in ranges for value in (item.start_ns, item.end_ns)]
    if not timestamps:
        draw.text((left + 8, top + 8), "No trace events", fill="#111827", font=font)
        return

    start_ns = min(timestamps)
    end_ns = max(timestamps)
    duration_ns = max(1, end_ns - start_ns)
    for tick in range(6):
        x = axis_left + (axis_right - axis_left) * tick / 5
        draw.line((x, top + title_height, x, bottom - 16), fill="#e5e7eb", width=1)
        label = f"{duration_ns * tick / 5000:.1f} us"
        draw.text((x - 14, bottom - 14), label, fill="#475569", font=font)

    rows = [(cta, track) for cta in selected_ctas for track in tracks]
    row_y = {(cta, track): top + title_height + index * row_height for index, (cta, track) in enumerate(rows)}
    for index, (cta, track) in enumerate(rows):
        y = row_y[(cta, track)]
        if index % 2 == 0:
            draw.rectangle((left, y, right, y + row_height - 1), fill="#f8fafc")
        draw.text((left + 4, y + 2), f"CTA {cta:03d} {track}", fill="#1f2937", font=font)

    for item in ranges:
        key = (item.cta, item.track)
        if key not in row_y:
            continue
        x0 = axis_left + (item.start_ns - start_ns) * (axis_right - axis_left) / duration_ns
        x1 = axis_left + (item.end_ns - start_ns) * (axis_right - axis_left) / duration_ns
        y0 = row_y[key] + 2
        y1 = y0 + max(5, row_height - 5)
        draw.rectangle((x0, y0, max(x0 + 1, x1), y1), fill=PHASE_COLORS[item.phase])


def save_pipeline_trace_png(
    l1_trace,
    l2_trace,
    output_path: str | Path,
    selected_ctas: Iterable[int],
    title: str,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    selected = tuple(dict.fromkeys(int(cta) for cta in selected_ctas))
    l1_ranges = _collect_l1(l1_trace, selected)
    l2_ranges = _collect_l2(l2_trace, selected)
    width = 1900
    rows = max(len(selected) * 4, len(selected) * 3)
    height = max(420, 112 + rows * 22)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((18, 12), title, fill="#111827", font=font)

    legend_x = 18
    for phase, color in PHASE_COLORS.items():
        draw.rectangle((legend_x, 34, legend_x + 12, 46), fill=color)
        draw.text((legend_x + 17, 34), phase, fill="#334155", font=font)
        legend_x += 17 + 7 * len(phase) + 18

    panel_top = 62
    panel_bottom = height - 8
    panel_mid = width // 2
    draw.text((18, panel_top), "L1: dispatch / TMA / GEMM / SwiGLU", fill="#111827", font=font)
    draw.text((panel_mid + 8, panel_top), "L2: TMA / GEMM / scatter / reduce", fill="#111827", font=font)
    _draw_panel(
        draw,
        l1_ranges,
        selected,
        ("dispatch0", "dispatch1", "producer", "math"),
        (12, panel_top + 18, panel_mid - 4, panel_bottom),
    )
    _draw_panel(draw, l2_ranges, selected, ("producer", "math", "reduce"), (panel_mid + 4, panel_top + 18, width - 12, panel_bottom))

    output = Path(output_path).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output
