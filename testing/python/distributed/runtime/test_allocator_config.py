"""Configuration-surface tests for the distributed allocator.

These cover the non-collective parts only, so they need one GPU at most.
"""

from __future__ import annotations

import os

import pytest

import tilelang.testing
from tilelang.distributed.allocator import BaseAllocator, _parse_bool_env, _resolve_use_vmm


@pytest.mark.parametrize("value", ["1", "true", "True", "ON", "yes", "y"])
def test_bool_env_accepts_truthy_spellings(value):
    assert _parse_bool_env("TILESCALE_USE_VMM", value) is True


@pytest.mark.parametrize("value", ["0", "false", "False", "off", "no", "n", ""])
def test_bool_env_accepts_falsy_spellings(value):
    assert _parse_bool_env("TILESCALE_USE_VMM", value) is False


@pytest.mark.parametrize("value", ["2", "maybe", "on/off"])
def test_bool_env_rejects_unreadable_values(value):
    """An unreadable value must not quietly resolve to False.

    `TILESCALE_USE_VMM=true` once disabled VMM, the opposite of the intent.
    """
    with pytest.raises(ValueError, match="must be a boolean value"):
        _parse_bool_env("TILESCALE_USE_VMM", value)


def test_use_vmm_env_overrides_argument():
    previous = os.environ.get("TILESCALE_USE_VMM")
    try:
        os.environ["TILESCALE_USE_VMM"] = "true"
        assert _resolve_use_vmm(False, is_distributed=True) is True
        os.environ["TILESCALE_USE_VMM"] = "off"
        assert _resolve_use_vmm(True, is_distributed=True) is False
    finally:
        if previous is None:
            os.environ.pop("TILESCALE_USE_VMM", None)
        else:
            os.environ["TILESCALE_USE_VMM"] = previous


@tilelang.testing.requires_cuda
def test_table_properties_on_non_distributed_allocator():
    """`table_size` must degrade like `table` instead of raising AttributeError."""
    allocator = BaseAllocator(1 << 20, device="cuda:0", is_distributed=False)
    try:
        assert allocator.table is None
        assert allocator.table_size == 0
    finally:
        allocator.close()


if __name__ == "__main__":
    tilelang.testing.main()
