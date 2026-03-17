"""The language interface for tl programs."""

"""This file provides interface for inter-node comm via IBGDA.
For now, we rely on NVSHMEM to implement IBGDA."""

from tvm import tir


def ibgda_get_qps_per_rdma_rank():
    return tir.call_intrin("int32", tir.op.Op.get("tl.IbgdaGetQpsPerRdmaRank"))


def ibgda_quiet(dst_pe, qp_id):
    return tir.call_intrin("handle", tir.op.Op.get("tl.IbgdaQuiet"), dst_pe, qp_id)


def ibgda_put_nbi_warp(req_rptr, req_lptr, bytes, dst_pe, qp_id, lane_id, message_idx, always_do_post_send=False):
    return tir.call_intrin("handle", tir.op.Op.get("tl.IbgdaPutNbiWarp"), req_rptr, req_lptr, bytes, dst_pe, qp_id, lane_id, message_idx, always_do_post_send)


def ibgda_amo_nonfetch_add(rptr, value, pe, qp_id, is_local_copy=False):
    return tir.call_intrin("handle", tir.op.Op.get("tl.IbgdaAmoNonfetchAdd"), rptr, value, pe, qp_id, is_local_copy)
