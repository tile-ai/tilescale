# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tvm import tir


def get_pe(*args):
    return tir.call_intrin("int32", tir.op.Op.get("tl.GetPE"), *args)


def get_pe_num(*args):
    return tir.call_intrin("int32", tir.op.Op.get("tl.GetPENum"), *args)


def int_p(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.IntPE"), *args)


def barrier_all(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.BarrierAll"), *args)


def barrier_all_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.BarrierAllBlock"), *args)


def barrier_all_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.BarrierAllWarp"), *args)


def sync_all(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.SyncAll"), *args)


def sync_all_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.SyncAllBlock"), *args)


def sync_all_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.SyncAllWarp"), *args)


def quiet(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.Quiet"), *args)


def fence(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.Fence"), *args)


def getmem_nbi_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetmemNbiBlock"), *args)


def getmem_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetmemBlock"), *args)


def getmem_nbi_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetmemNbiWarp"), *args)


def getmem_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetmemWarp"), *args)


def getmem_nbi(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.GetmemNbi"), *args)


def getmem(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.Getmem"), *args)


def putmem_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemBlock"), *args)


def putmem_nbi_block(*args):
    return tir.call_intrin("int32", tir.op.Op.get("tl.PutmemNbiBlock"), *args)


def putmem_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemWarp"), *args)


def putmem_nbi_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemNbiWarp"), *args)


def putmem(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.Putmem"), *args)


def putmem_nbi(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemNbi"), *args)


def putmem_signal(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignal"), *args)


def putmem_signal_nbi(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignalNbi"), *args)


def putmem_signal_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignalBlock"), *args)


def putmem_signal_nbi_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignalNbiBlock"), *args)


def putmem_signal_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignalWarp"), *args)


def putmem_signal_nbi_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.PutmemSignalNbiWarp"), *args)


def signal_op(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.SignalOp"), *args)


def signal_wait_until(*args):
    return tir.call_intrin("int32", tir.op.Op.get("tl.SignalWaitUntil"), *args)


def broadcast(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.Broadcast"), *args)


def broadcast_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.BroadcastWarp"), *args)


def broadcast_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.BroadcastBlock"), *args)


def broadcastmem_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.BroadcastmemBlock"), *args)


def fcollect(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.Fcollect"), *args)


def fcollect_warp(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.FcollectWarp"), *args)


def fcollect_block(*args):
    return tir.call_intrin("handle", tir.op.Op.get("tl.FcollectBlock"), *args)
