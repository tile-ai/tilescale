# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm import tir


def get_pe(*args):
    return tir.call_intrin("int32", tir.op.Op.get("tl.GetPE"), *args)