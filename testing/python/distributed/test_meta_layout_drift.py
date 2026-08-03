"""Guard the metadata table layout against Python/C++ drift.

The peer pointer table is written by tilelang/distributed/allocator.py and read
by src/cuda/runtime.cc and the device headers. The offsets are declared twice --
once as ``_META_*`` Python ints, once as ``TL_META_*`` C preprocessor defines --
so a one-sided edit silently corrupts remote address computation instead of
failing loudly. These tests parse both declarations and compare them.
"""

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PY_SOURCE = _REPO_ROOT / "tilelang/distributed/allocator.py"
_H_SOURCE = _REPO_ROOT / "src/tl_templates/cuda/distributed/meta_layout.h"

_PY_RE = re.compile(r"^_META_([A-Z0-9_]+)\s*=\s*(\d+)\s*$", re.MULTILINE)
# The header mixes numeric defines with aliases (HEADER_SIZE -> PEER_BASE), so
# resolve one level of TL_META_* indirection rather than only literals.
_H_RE = re.compile(
    r"^#define\s+TL_META_([A-Z0-9_]+)\s+(\d+|TL_META_[A-Z0-9_]+)\s*$", re.MULTILINE)


def _parse(path: Path, pattern: re.Pattern) -> dict[str, int]:
    return {m.group(1): int(m.group(2)) for m in pattern.finditer(path.read_text())}


def _parse_header(path: Path) -> dict[str, int]:
    raw = {m.group(1): m.group(2) for m in _H_RE.finditer(path.read_text())}
    resolved = {k: int(v) for k, v in raw.items() if v.isdigit()}
    for key, value in raw.items():
        if key not in resolved:
            target = value.removeprefix("TL_META_")
            assert target in resolved, f"TL_META_{key} aliases unresolvable {value}"
            resolved[key] = resolved[target]
    return resolved


def test_sources_exist():
    assert _PY_SOURCE.is_file(), f"missing {_PY_SOURCE}"
    assert _H_SOURCE.is_file(), f"missing {_H_SOURCE}"


def test_layout_constants_match():
    py = _parse(_PY_SOURCE, _PY_RE)
    hdr = _parse_header(_H_SOURCE)

    assert py, "no _META_* constants parsed from allocator.py"
    assert hdr, "no TL_META_* defines parsed from meta_layout.h"

    # HEADER_SIZE is a derived alias in the header with no Python counterpart.
    hdr_offsets = {k: v for k, v in hdr.items() if k != "HEADER_SIZE"}

    assert hdr_offsets.keys() == py.keys(), (
        "metadata field sets diverged; "
        f"python-only={sorted(py.keys() - hdr_offsets.keys())} "
        f"header-only={sorted(hdr_offsets.keys() - py.keys())}"
    )

    mismatched = {k: (py[k], hdr_offsets[k]) for k in py if py[k] != hdr_offsets[k]}
    assert not mismatched, f"offset mismatch (python, header): {mismatched}"


def test_offsets_are_a_dense_unique_range():
    """A gap or duplicate offset means two fields alias or a slot is unread."""
    py = _parse(_PY_SOURCE, _PY_RE)
    offsets = sorted(py.values())
    assert offsets == list(range(len(offsets))), (
        f"offsets are not a dense 0..N-1 range: {sorted(py.items(), key=lambda kv: kv[1])}"
    )


def test_peer_base_is_last_and_matches_header_size():
    """Peer pointers are variable-length, so they must occupy the tail."""
    py = _parse(_PY_SOURCE, _PY_RE)
    hdr = _parse_header(_H_SOURCE)

    assert py["PEER_BASE"] == max(py.values()), (
        "PEER_BASE must be the final field; the per-rank pointer array is "
        "appended after it and would otherwise overwrite scalar metadata"
    )
    assert hdr["HEADER_SIZE"] == hdr["PEER_BASE"], (
        "TL_META_HEADER_SIZE must equal TL_META_PEER_BASE so the scalar header "
        "ends exactly where the peer pointer array begins"
    )


def test_runtime_uses_named_peer_base_offset():
    """Regression guard: runtime.cc previously hardcoded ``meta_data[2 + rank]``.

    After the header grew, that raw index pointed at scalar fields instead of
    the peer pointer array, producing wrong remote TMA addresses.
    """
    runtime_cc = _REPO_ROOT / "src/cuda/runtime.cc"
    text = runtime_cc.read_text()
    raw_indices = re.findall(r"meta_data\[\s*\d+\s*[+\]]", text)
    assert not raw_indices, (
        f"{runtime_cc.name} indexes the metadata table with raw integers "
        f"{raw_indices}; use the TL_META_* constants so the offsets track "
        "meta_layout.h"
    )
