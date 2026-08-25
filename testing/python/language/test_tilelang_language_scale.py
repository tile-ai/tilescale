import torch

import tilelang
import tilelang.language as T
import tilelang.testing
import tilelang.transform
import pytest

_HAS_EXPANSION = hasattr(tilelang.transform, "NormalizeScaleExpansion")
requires_expansion = pytest.mark.skipif(
    not _HAS_EXPANSION, reason="NormalizeScaleExpansion (M2) not ported yet")


@tilelang.testing.requires_cuda
def test_scale_lowering_from_prim_func():
    @T.prim_func
    def kernel(schedule: T.Tensor((148,), T.int32), out: T.Tensor((148,), T.int32)):
        with T.Scale("die", 2) as did:
            with T.Scale("sm-cluster", 2, cluster_size=2) as cid:
                with T.Scale("sm", 37, num_sms_per_die=74, cluster_size=2, sm_schedule=schedule) as bid:
                    with T.Scale("thread", 256) as tid:
                        if tid == 0:
                            out[bid * 4 + did * 2 + cid] = did * 1000 + bid * 10 + cid

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source

    assert "threadIdx.x" in source
    assert "tl::block_rank_in_cluster()" in source
    assert "tl::get_smid()" in source
    device_func = next(iter(artifact.device_mod.functions.values()))
    assert list(device_func.attrs["cluster_dims"]) == [2, 1, 1]


@tilelang.testing.requires_cuda
def test_scale_lowering_from_eager_jit():
    @tilelang.jit
    def kernel(schedule: T.Tensor((148,), "int32")):
        out = T.empty((148,), dtype="int32")
        with T.Scale("die", 2) as did:
            with T.Scale("sm-cluster", 2, cluster_size=2) as cid:
                with T.Scale("sm", 37, num_sms_per_die=74, cluster_size=2, sm_schedule=schedule) as bid:
                    with T.Scale("thread", 256) as tid:
                        if tid == 0:
                            out[bid * 4 + did * 2 + cid] = did * 1000 + bid * 10 + cid
        return out

    schedule = torch.arange(148, dtype=torch.int32, device="cuda")
    source = kernel.get_kernel_source(schedule)

    assert "threadIdx.x" in source
    assert "tl::block_rank_in_cluster()" in source
    assert "tl::get_smid()" in source


@tilelang.testing.requires_cuda
def test_scale_device():
    @T.prim_func
    def kernel(out: T.Tensor((4,), T.int32)):
        with T.Scale("device"):
            with T.Scale("block", 4) as bx:
                with T.Scale("thread", 32) as tid:
                    if tid == 0:
                        out[bx] = bx

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


@tilelang.testing.requires_cuda
def test_scale_device_multi_rank_lowering():
    """T.Scale('device', n>1) binds the rank var to tl::get_rank() (SPMD)."""

    @T.prim_func
    def kernel(out: T.Tensor((8,), T.int32)):
        with T.Scale("device", 8) as di:
            with T.Scale("block", 1) as _bx:
                with T.Scale("thread", 32) as tid:
                    if tid == 0:
                        out[di] = di

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "tl::get_rank()" in source


@tilelang.testing.requires_cuda
def test_scale_block_2d():
    M, N = 64, 128
    bM, bN = 16, 32

    @T.prim_func
    def kernel(out: T.Tensor((M, N), T.int32)):
        with T.Scale("block", T.ceildiv(N, bN), T.ceildiv(M, bM)) as (bx, by):
            with T.Scale("thread", 32) as tid:
                if tid == 0:
                    out[by * bM, bx * bN] = by * 100 + bx

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


@tilelang.testing.requires_cuda
def test_scale_warp_in_block():
    @T.prim_func
    def kernel(out: T.Tensor((256,), T.int32)):
        with T.Scale("block", 4) as bx:
            with T.Scale("warp", 4) as wx:
                out[bx * 64 + wx * 16] = wx

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


@tilelang.testing.requires_cuda
def test_scale_cluster_outer():
    # cluster gives grid coords, block gives the CTA rank within the cluster.
    @T.prim_func
    def kernel(out: T.Tensor((16,), T.int32)):
        with T.Scale("cluster", 4, 2) as (cx, cy):
            with T.Scale("block", 2) as bid:
                with T.Scale("thread", 32) as tid:
                    if tid == 0:
                        out[(cx * 2 + bid)] = cy

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    assert "block_rank_in_cluster" in source
    device_func = next(iter(artifact.device_mod.functions.values()))
    assert list(device_func.attrs["cluster_dims"]) == [2, 1, 1]


@tilelang.testing.requires_cuda
def test_scale_swizzle():
    # swizzle= must compose with Scale launch loops (no free vars / dropped loops).
    @T.prim_func
    def kernel(out: T.Tensor((64, 128), T.int32)):
        with T.Scale("block", T.ceildiv(128, 32), T.ceildiv(64, 16), swizzle=8) as (bx, by):
            with T.Scale("thread", 32) as tid:
                if tid == 0:
                    out[by * 16, bx * 32] = by * 100 + bx

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    assert "rasterization2DRow" in source


@tilelang.testing.requires_cuda
def test_scale_swizzle_block_warp():
    # Regression for the example_gemm_intrinsics structure:
    # T.Scale("block", swizzle=10) -> T.Scale("warp") must lower without crashing
    # (swizzle attr composes with the warp scale's threadIdx hoisting).
    @T.prim_func
    def kernel(out: T.Tensor((128, 128), T.int32)):
        with T.Scale("block", T.ceildiv(128, 64), T.ceildiv(128, 64), swizzle=10) as (bx, by):
            with T.Scale("warp", 4) as wx:
                if wx == 0:
                    out[by * 64, bx * 64] = by * 100 + bx

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    assert "rasterization2DRow" in source


@tilelang.testing.requires_cuda
def test_scale_workgroup_api():
    # Lowercase T.scale with workgroup= shape, the preferred public spelling.
    @T.prim_func
    def kernel(out: T.Tensor((64, 128), T.int32)):
        with T.scale("block", workgroup=(4, 2)) as (bx, by):
            with T.scale("thread", workgroup=(32,)) as tid:
                if tid == 0:
                    out[by * 16, bx * 32] = by * 100 + bx

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


@tilelang.testing.requires_cuda
def test_scale_workgroup_matches_positional():
    # workgroup= and the legacy positional form must lower equivalently.
    # Compare structurally on the launch shape (thread_extent of every
    # blockIdx/threadIdx binding) rather than full-source string equality.
    import tvm
    from tvm import tirx as tir

    def build(use_workgroup):
        @T.prim_func
        def kernel(out: T.Tensor((64, 128), T.int32)):
            if use_workgroup:
                block_cm = T.scale("block", workgroup=(4, 2))
                thread_cm = T.scale("thread", workgroup=(32,))
            else:
                block_cm = T.Scale("block", 4, 2)
                thread_cm = T.scale("thread", 32)
            with block_cm as (bx, by):
                with thread_cm as tid:
                    if tid == 0:
                        out[by * 16, bx * 32] = by * 100 + bx

        mod = tvm.IRModule.from_expr(kernel)
        return tilelang.transform.LowerScaleLaunch()(mod)

    def launch_shape(mod):
        # Map each thread tag (blockIdx.x/y, threadIdx.x/...) to its extent.
        shape = {}

        def visit(stmt):
            if isinstance(stmt, tir.AttrStmt) and stmt.attr_key == "thread_extent":
                iv = stmt.node
                ext = stmt.value
                shape[str(iv.thread_tag)] = int(ext) if isinstance(ext, (tir.IntImm,)) else ext

        func = next(iter(mod.functions.values()))
        tir.stmt_functor.post_order_visit(func.body, visit)
        return shape

    sw = launch_shape(build(True))
    pos = launch_shape(build(False))
    assert sw == pos, f"workgroup {sw} != positional {pos}"
    # Sanity: the actual launch dims, not just presence.
    assert sw.get("blockIdx.x") == 4 and sw.get("blockIdx.y") == 2
    assert sw.get("threadIdx.x") == 32


@tilelang.testing.requires_cuda
def test_scale_block_workgroup_3d_cta_rank():
    # The plan's 2CTA form: cluster grid + block(workgroup=(2,1,1)) as a single
    # CTA rank. cluster_dims must be (2,1,1) and the rank var unpacks as a scalar.
    @T.prim_func
    def kernel(out: T.Tensor((16,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                with T.scale("thread", workgroup=(32,)) as tid:
                    if tid == 0:
                        out[cm * 2 + cta_id] = cn

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    assert "block_rank_in_cluster" in source
    device_func = next(iter(artifact.device_mod.functions.values()))
    assert list(device_func.attrs["cluster_dims"]) == [2, 1, 1]


def test_scale_workgroup_and_positional_conflict():
    # Passing both positional extents and workgroup= is an error.
    import pytest

    def build():
        @T.prim_func
        def kernel(out: T.Tensor((4,), T.int32)):
            with T.scale("block", 4, workgroup=(4,)) as bx:
                with T.scale("thread", workgroup=(32,)) as tid:
                    if tid == 0:
                        out[bx] = bx

        return kernel

    with pytest.raises(Exception):
        build()


def test_scale_no_grid_kwarg():
    # The model intentionally has no grid= parameter.
    import inspect
    sig = inspect.signature(T.scale)
    assert "grid" not in sig.parameters
    assert "workgroup" in sig.parameters
    assert "grid" not in inspect.signature(T.Scale).parameters


@tilelang.testing.requires_cuda
def test_scale_workgroup_scalar_and_list():
    # workgroup= accepts an int scalar and a list, normalized to extents.
    @T.prim_func
    def kernel(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=4) as bx:          # scalar int
            with T.scale("thread", workgroup=[32]) as tid:  # list
                if tid == 0:
                    out[bx * 32] = bx

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


def test_scale_empty_workgroup_rejected():
    # An explicit empty workgroup= shape is an error (no silent extent-1).
    import pytest

    def build():
        @T.prim_func
        def kernel(out: T.Tensor((4,), T.int32)):
            with T.scale("device", workgroup=()) as d:
                with T.scale("thread", workgroup=(32,)) as tid:
                    if tid == 0:
                        out[0] = d

        return kernel

    with pytest.raises(Exception):
        build()


@tilelang.testing.requires_cuda
def test_scale_non_scale_stmt_between_scales_rejected():
    # Statements / control flow between a *cluster* and its inner block are only
    # supported for no-op statements (T.assume / T.serial) as of Step 2B; an
    # `if` or an execution-affecting statement on that edge is still rejected
    # with a clear, specific error (rather than segfault or silent miscompile).
    #
    # NOTE: Step 1A/1B support interleaving in the block-to-thread gap, and
    # Step 2B supports cluster -> {T.assume/T.serial} -> block. Those are
    # exercised as positive tests below, not here.
    import pytest

    def expect_reject(fn):
        with pytest.raises(Exception) as ei:
            tilelang.lower(fn, target="cuda")
        msg = str(ei.value)
        # Must be our specific guard, not an unrelated lowering crash.
        assert ("not supported yet" in msg
                and "scale" in msg), f"unexpected error: {msg[:200]}"

    # Inner block wrapped in control flow (an `if`) under a cluster: the
    # cluster-to-block edge stays strict for control flow.
    @T.prim_func
    def k_if(out: T.Tensor((16,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            if cm >= 0:
                with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                    with T.scale("thread", workgroup=(32,)) as tid:
                        if tid == 0:
                            out[cm * 2 + cta_id] = cn

    expect_reject(k_if)

    # cluster -> store -> block: an execution-affecting buffer store on the
    # cluster-to-block edge would run per-CTA; rejected.
    @T.prim_func
    def k_store(out: T.Tensor((16,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            out[cm] = cn
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                with T.scale("thread", workgroup=(32,)) as tid:
                    if tid == 0:
                        out[cm * 2 + cta_id] = cn

    expect_reject(k_store)

    # cluster -> T.copy -> block: a tile op on the cluster-to-block edge.
    @T.prim_func
    def k_copy(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            smem = T.alloc_shared((64, 64), T.float16)
            T.copy(A[cm * 16, 0], smem)
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                with T.scale("thread", workgroup=(32,)) as tid:
                    B[(cm * 2 + cta_id) * 16 + tid, 0] = smem[0, 0]

    expect_reject(k_copy)

    # cluster -> two block scales: the bridge requires a single block target.
    @T.prim_func
    def k_two_blocks(out: T.Tensor((16,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cid0:
                with T.scale("thread", workgroup=(32,)) as t0:
                    if t0 == 0:
                        out[cm * 2 + cid0] = cn
            with T.scale("block", workgroup=(2, 1, 1)) as cid1:
                with T.scale("thread", workgroup=(32,)) as t1:
                    if t1 == 0:
                        out[cm * 2 + cid1] = cn

    expect_reject(k_two_blocks)


@tilelang.testing.requires_cuda
def test_scale_block_serial_thread_allowed():
    # Step 1A: a non-scale loop (T.serial) between a plain block scale and its
    # inner thread scale must now lower successfully (not be rejected). The
    # statement stays in place; the thread binding is owned by the block.
    # Using `tid` in the store proves the standalone thread var was actually
    # bound to threadIdx.x (an unbound var would crash lowering).
    @T.prim_func
    def kernel(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            for _i in T.serial(1):
                with T.scale("thread", workgroup=(32,)) as tid:
                    out[bx * 32 + tid] = tid

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


@tilelang.testing.requires_cuda
def test_scale_block_assume_thread_allowed():
    # Step 1A: a T.assume between a plain block scale and its inner thread scale
    # must lower successfully, with the statement kept in place. `tid` is used to
    # confirm the thread var is bound.
    @T.prim_func
    def kernel(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            T.assume(bx >= 0)
            with T.scale("thread", workgroup=(32,)) as tid:
                out[bx * 32 + tid] = tid

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


@tilelang.testing.requires_cuda
def test_scale_block_seqstmt_thread_allowed():
    # Step 1A: a thread scale buried inside a block body's SeqStmt, with multiple
    # no-op T.assume statements before it and a pass-through serial loop, must
    # lower successfully. `tid` is used to confirm the thread var is bound.
    @T.prim_func
    def kernel(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            T.assume(bx >= 0)
            T.assume(bx < 4)
            for _i in T.serial(1):
                with T.scale("thread", workgroup=(32,)) as tid:
                    out[bx * 32 + tid] = tid

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


@tilelang.testing.requires_cuda
def test_scale_block_interleave_rejects():
    # Step 1A explicitly does NOT support execution-affecting statements between
    # a block scale and its inner thread scale, nor more than one hidden thread
    # scale. These must raise the specific Step 1A guard, not silently
    # miscompile (the intervening statement would otherwise run per-thread).
    import pytest

    def expect_reject(fn):
        with pytest.raises(Exception) as ei:
            tilelang.lower(fn, target="cuda")
        msg = str(ei.value)
        assert ("not supported yet" in msg
                and "scale" in msg), f"unexpected error: {msg[:200]}"

    # block -> store -> thread: a buffer store between the levels would run
    # per-thread under the lifted threadIdx binding.
    @T.prim_func
    def k_store(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            out[bx * 32] = bx
            with T.scale("thread", workgroup=(32,)) as tid:
                out[bx * 32 + tid] = tid

    expect_reject(k_store)

    # block -> if -> thread: control flow wrapping the inner scale.
    @T.prim_func
    def k_if(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            if bx >= 0:
                with T.scale("thread", workgroup=(32,)) as tid:
                    out[bx * 32 + tid] = tid

    expect_reject(k_if)

    # block -> two thread scales: only one deferral slot exists; a second hidden
    # thread scale must be rejected, not silently drop the first binding.
    @T.prim_func
    def k_two_threads(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            for _i in T.serial(1):
                with T.scale("thread", workgroup=(32,)) as tid0:
                    out[bx * 32 + tid0] = tid0
            for _j in T.serial(1):
                with T.scale("thread", workgroup=(32,)) as tid1:
                    out[bx * 32 + tid1] = tid1

    expect_reject(k_two_threads)

    # block -> serial -> thread -> block: the hidden thread is a terminal scale;
    # nesting a block scale below it must be rejected, not silently dropped.
    @T.prim_func
    def k_thread_then_block(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            for _i in T.serial(1):
                with T.scale("thread", workgroup=(32,)) as tid:
                    with T.scale("block", workgroup=(2,)) as bx2:
                        out[bx * 32 + tid] = bx2

    expect_reject(k_thread_then_block)

    # block -> serial -> thread -> cluster: likewise, a cluster scale below the
    # terminal thread scale must be rejected.
    @T.prim_func
    def k_thread_then_cluster(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            for _i in T.serial(1):
                with T.scale("thread", workgroup=(32,)) as tid:
                    with T.scale("cluster", workgroup=(2,)) as cx:
                        out[bx * 32 + tid] = cx

    expect_reject(k_thread_then_cluster)

    # block -> thread -> block (direct, no interleaving): the thread scale must
    # be terminal; a block nested directly below it must be rejected by the
    # terminal-scale check rather than mis-lowered as an extra grid dimension.
    @T.prim_func
    def k_thread_then_block_direct(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            with T.scale("thread", workgroup=(32,)) as tid:
                with T.scale("block", workgroup=(2,)) as bx2:
                    out[bx * 32 + tid] = bx2

    expect_reject(k_thread_then_block_direct)


@tilelang.testing.requires_cuda
@requires_expansion
def test_scale_block_copy_thread_allowed():
    # block -> T.copy -> thread now lowers via the generic top-down expansion
    # path: the ("block", "tl.tileop.copy") ScaleExpansionTemplate plans a
    # block->thread expansion, and NormalizeScaleExpansion merges the cooperative
    # copy into the thread region with a CTA sync before the thread read. This is
    # NOT a special-case pass (see test_scale_expansion_template_resolves_block_copy
    # for the dispatch proof).
    import tvm

    @T.prim_func
    def k_copy(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            smem = T.alloc_shared((64, 64), T.float16)
            T.copy(A[bx, 0, 0], smem)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[bx, tid] = smem[tid, 0]

    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(k_copy, target=target)
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    assert "scale_ctx" not in source
    # The cooperative copy writes shared before the thread read -- a CTA sync
    # (inserted by the expansion plan's sync_threads BarrierSpec) must sit between.
    assert ("__syncthreads" in source) or ("tvm_storage_sync" in source), \
        "expected a CTA sync between the block-scope copy and the thread read"


@tilelang.testing.requires_cuda
@requires_expansion
def test_scale_block_copy_thread_correct():
    # GPU correctness for the generic block-copy expansion: B[bx, tid] == A[bx, tid, 0].
    import torch

    @T.prim_func
    def k_copy(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            smem = T.alloc_shared((64, 64), T.float16)
            T.copy(A[bx, 0, 0], smem)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[bx, tid] = smem[tid, 0]

    kernel = tilelang.compile(k_copy, target="cuda")
    A = torch.randn(4, 64, 64, device="cuda", dtype=torch.float16)
    B = torch.empty(4, 32, device="cuda", dtype=torch.float16)
    kernel(A, B)
    torch.testing.assert_close(B, A[:, :32, 0])


@tilelang.testing.requires_cuda
@requires_expansion
def test_scale_block_fill_thread_allowed():
    # Milestone 13 MVP: a block-scope elementwise T.fill before the inner thread
    # scale lowers via the generic top-down path (BlockFillExpansionTemplate), the
    # same machinery as block-copy. The fill writes shared, so the generic barrier
    # planner inserts a CTA sync before the thread read. Not a special case in the
    # normalizer (see test_scale_expansion_template_resolves_block_fill).
    import tvm

    @T.prim_func
    def k_fill(B: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            smem = T.alloc_shared((64, 64), T.float16)
            T.fill(smem, 0)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[bx, tid] = smem[tid, 0]

    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(k_fill, target=target)
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    assert "scale_ctx" not in source
    assert ("__syncthreads" in source) or ("tvm_storage_sync" in source), \
        "expected a CTA sync between the block-scope fill (shared) and the read"


@tilelang.testing.requires_cuda
@requires_expansion
def test_scale_block_fill_thread_correct():
    # GPU correctness for the generic block-fill expansion: B is all zeros.
    import torch

    @T.prim_func
    def k_fill(B: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            smem = T.alloc_shared((64, 64), T.float16)
            T.fill(smem, 0)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[bx, tid] = smem[tid, 0]

    kernel = tilelang.compile(k_fill, target="cuda")
    B = torch.empty(4, 32, device="cuda", dtype=torch.float16).fill_(7.0)
    kernel(B)
    torch.testing.assert_close(B, torch.zeros(4, 32, device="cuda", dtype=torch.float16))


@requires_expansion
def test_scale_copy_effects_scope_accurate():
    # Milestone 14: the block-copy template reports effects from the actual
    # src/dst storage scopes (not a hard-coded global->shared assumption). A copy
    # into shared reports writes_shared (block sync derivable); a copy into global
    # reports writes_global (no block sync). No normalizer change -- pure template
    # effects accuracy.
    import tvm
    from tilelang.transform import plan_region_expansions

    @T.prim_func
    def g2s(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            smem = T.alloc_shared((64, 64), T.float16)
            T.copy(A[bx, 0, 0], smem)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[bx, tid] = smem[tid, 0]

    @T.prim_func
    def g2g(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 64, 64), T.float16),
            Out: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            T.copy(A[bx, 0:64, 0:64], B[bx, 0:64, 0:64])
            with T.scale("thread", workgroup=(32,)) as tid:
                Out[bx, tid] = B[bx, tid, 0]

    m1 = tvm.IRModule.from_expr(g2s.with_attr("global_symbol", "main"))
    e1 = plan_region_expansions(m1["main"])[0].effects
    assert e1.reads_global is True and e1.writes_shared is True
    assert e1.writes_global is False

    m2 = tvm.IRModule.from_expr(g2g.with_attr("global_symbol", "main"))
    e2 = plan_region_expansions(m2["main"])[0].effects
    assert e2.reads_global is True and e2.writes_global is True
    assert e2.writes_shared is False  # no spurious shared write -> no block sync


@tilelang.testing.requires_cuda
@requires_expansion
def test_scale_block_copy_global_to_global_allowed():
    # Broadened copy direction (milestone 14): a block-scope global->global tile
    # copy before the inner thread scale lowers via the same generic path. The
    # destination is global (not shared), so the planner derives NO block sync.
    import tvm

    @T.prim_func
    def k(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 64, 64), T.float16),
          Out: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            T.copy(A[bx, 0:64, 0:64], B[bx, 0:64, 0:64])
            with T.scale("thread", workgroup=(32,)) as tid:
                Out[bx, tid] = B[bx, tid, 0]

    target = tvm.target.Target("cuda")
    with target:
        artifact = tilelang.lower(k, target=target)
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    assert "scale_ctx" not in source


@tilelang.testing.requires_cuda
@requires_expansion
def test_scale_block_copy_global_to_global_correct():
    # GPU correctness for the broadened global->global block copy: Out == A[:, :32, 0].
    import torch

    @T.prim_func
    def k(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 64, 64), T.float16),
          Out: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            T.copy(A[bx, 0:64, 0:64], B[bx, 0:64, 0:64])
            with T.scale("thread", workgroup=(32,)) as tid:
                Out[bx, tid] = B[bx, tid, 0]

    kernel = tilelang.compile(k, target="cuda")
    A = torch.randn(4, 64, 64, device="cuda", dtype=torch.float16)
    B = torch.empty(4, 64, 64, device="cuda", dtype=torch.float16)
    Out = torch.empty(4, 32, device="cuda", dtype=torch.float16)
    kernel(A, B, Out)
    torch.testing.assert_close(Out, A[:, :32, 0])


@tilelang.testing.requires_cuda
def test_scale_block_serial_warp_allowed():
    # Step 1A supports a hidden *warp* scale under a block too (the helper allows
    # warp). A warp scale behind a T.serial must lower successfully; `wid` is
    # used so an unbound warp var would crash lowering.
    @T.prim_func
    def kernel(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            for _i in T.serial(1):
                with T.scale("warp", workgroup=(4,)) as wid:
                    out[bx * 4 + wid] = wid

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


@tilelang.testing.requires_cuda
def test_scale_block_alloc_thread_allowed():
    # A T.alloc_shared in the block scope is hoisted to the root block as a
    # block/root-scoped allocation; it does NOT become a statement between the
    # block and thread levels (the two scales end up directly nested). So this
    # lowers successfully -- there is nothing per-thread about the allocation.
    @T.prim_func
    def kernel(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            smem = T.alloc_shared((4,), T.int32)
            with T.scale("thread", workgroup=(32,)) as tid:
                smem[0] = bx
                out[bx * 32 + tid] = smem[0]

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


def _assert_cluster_block_thread_interleave(artifact):
    """Shared asserts for Step 1B cluster -> block -> ... -> thread/warp kernels."""
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    # The block is the cluster-internal CTA rank.
    assert "block_rank_in_cluster" in source
    device_func = next(iter(artifact.device_mod.functions.values()))
    assert list(device_func.attrs["cluster_dims"]) == [2, 1, 1]


@tilelang.testing.requires_cuda
def test_scale_cluster_block_assume_thread_allowed():
    # Step 1B: cluster -> block(2,1,1) -> T.assume -> thread. The cluster and
    # block lower together (cluster_dims from the block workgroup); the hidden
    # thread is lifted to the block's launch group. The T.assume stays in place.
    @T.prim_func
    def kernel(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                T.assume(cta_id >= 0)
                with T.scale("thread", workgroup=(32,)) as tid:
                    out[(cm * 2 + cta_id) * 32 + tid] = cn + tid

    _assert_cluster_block_thread_interleave(tilelang.lower(kernel, target="cuda"))


@tilelang.testing.requires_cuda
def test_scale_cluster_block_serial_thread_allowed():
    # Step 1B: cluster -> block(2,1,1) -> T.serial -> thread.
    @T.prim_func
    def kernel(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                for _i in T.serial(1):
                    with T.scale("thread", workgroup=(32,)) as tid:
                        out[(cm * 2 + cta_id) * 32 + tid] = cn + tid

    _assert_cluster_block_thread_interleave(tilelang.lower(kernel, target="cuda"))


@tilelang.testing.requires_cuda
def test_scale_cluster_block_seqstmt_thread_allowed():
    # Step 1B: cluster -> block(2,1,1) -> SeqStmt(assume; assume; serial) ->
    # thread. Multiple no-op siblings before the hidden thread scale.
    @T.prim_func
    def kernel(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                T.assume(cta_id >= 0)
                T.assume(cta_id < 2)
                for _i in T.serial(1):
                    with T.scale("thread", workgroup=(32,)) as tid:
                        out[(cm * 2 + cta_id) * 32 + tid] = cn + tid

    _assert_cluster_block_thread_interleave(tilelang.lower(kernel, target="cuda"))


@tilelang.testing.requires_cuda
def test_scale_cluster_block_serial_warp_allowed():
    # Step 1B: cluster -> block(2,1,1) -> T.serial -> warp. The warp variant is
    # supported by the same helper; `wid` is used so an unbound warp var would
    # crash lowering.
    @T.prim_func
    def kernel(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                for _i in T.serial(1):
                    with T.scale("warp", workgroup=(4,)) as wid:
                        out[(cm * 2 + cta_id) * 4 + wid] = cn + wid

    _assert_cluster_block_thread_interleave(tilelang.lower(kernel, target="cuda"))


@tilelang.testing.requires_cuda
def test_scale_cluster_assume_block_thread_allowed():
    # Step 2B: cluster -> T.assume -> block(2,1,1) -> thread. A no-op T.assume on
    # the cluster-to-block edge is now bridged; cluster and block still lower
    # together (cluster_dims from the block workgroup) and the assume stays in
    # place. `cm` is referenced inside the assume to confirm the cluster rank var
    # is substituted there too.
    @T.prim_func
    def kernel(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            T.assume(cm >= 0)
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                with T.scale("thread", workgroup=(32,)) as tid:
                    out[(cm * 2 + cta_id) * 32 + tid] = cn + tid

    _assert_cluster_block_thread_interleave(tilelang.lower(kernel, target="cuda"))


@tilelang.testing.requires_cuda
def test_scale_cluster_serial_block_thread_allowed():
    # Step 2B: cluster -> T.serial -> block(2,1,1) -> thread. A pass-through
    # serial loop on the cluster-to-block edge is bridged.
    @T.prim_func
    def kernel(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            for _i in T.serial(1):
                with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                    with T.scale("thread", workgroup=(32,)) as tid:
                        out[(cm * 2 + cta_id) * 32 + tid] = cn + tid

    _assert_cluster_block_thread_interleave(tilelang.lower(kernel, target="cuda"))


@tilelang.testing.requires_cuda
def test_scale_cluster_seqstmt_block_thread_allowed():
    # Step 2B: cluster -> SeqStmt(assume; serial) -> block(2,1,1) -> thread.
    # Multiple no-op constructs on the cluster-to-block edge, kept in order.
    @T.prim_func
    def kernel(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            T.assume(cm >= 0)
            for _i in T.serial(1):
                with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                    with T.scale("thread", workgroup=(32,)) as tid:
                        out[(cm * 2 + cta_id) * 32 + tid] = cn + tid

    _assert_cluster_block_thread_interleave(tilelang.lower(kernel, target="cuda"))


@tilelang.testing.requires_cuda
def test_scale_cluster_assume_block_warp_allowed():
    # Step 2B: cluster -> T.assume -> block(2,1,1) -> warp. The warp variant is
    # bridged on the cluster-to-block edge too.
    @T.prim_func
    def kernel(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            T.assume(cm >= 0)
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                with T.scale("warp", workgroup=(4,)) as wid:
                    out[(cm * 2 + cta_id) * 4 + wid] = cn + wid

    _assert_cluster_block_thread_interleave(tilelang.lower(kernel, target="cuda"))


@tilelang.testing.requires_cuda
def test_scale_cluster_block_interleave_rejects():
    # The block-to-thread gap under a cluster-rank block is interleavable for
    # no-op statements only; it may not contain execution-affecting statements,
    # control flow, or more than one thread scale.
    import pytest

    def expect_reject(fn):
        with pytest.raises(Exception) as ei:
            tilelang.lower(fn, target="cuda")
        msg = str(ei.value)
        assert ("not supported yet" in msg
                and "scale" in msg), f"unexpected error: {msg[:200]}"

    # cluster -> block -> store -> thread: a buffer store between block and thread
    # would run per-thread under the lifted threadIdx binding.
    @T.prim_func
    def k_store(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                out[cm * 2 + cta_id] = cn
                with T.scale("thread", workgroup=(32,)) as tid:
                    out[(cm * 2 + cta_id) * 32 + tid] = tid

    expect_reject(k_store)

    # cluster -> block -> T.copy -> thread: T.copy is an execution-affecting tile
    # op between the levels.
    @T.prim_func
    def k_copy(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                smem = T.alloc_shared((64, 64), T.float16)
                T.copy(A[(cm * 2 + cta_id) * 16, 0], smem)
                with T.scale("thread", workgroup=(32,)) as tid:
                    B[(cm * 2 + cta_id) * 16 + tid, 0] = smem[0, 0]

    expect_reject(k_copy)

    # cluster -> block -> if -> thread: control flow wrapping the inner scale.
    @T.prim_func
    def k_if(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                if cta_id >= 0:
                    with T.scale("thread", workgroup=(32,)) as tid:
                        out[(cm * 2 + cta_id) * 32 + tid] = tid

    expect_reject(k_if)

    # cluster -> block -> two thread scales: only one deferral slot exists.
    @T.prim_func
    def k_two_threads(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                for _i in T.serial(1):
                    with T.scale("thread", workgroup=(32,)) as tid0:
                        out[(cm * 2 + cta_id) * 32 + tid0] = tid0
                for _j in T.serial(1):
                    with T.scale("thread", workgroup=(32,)) as tid1:
                        out[(cm * 2 + cta_id) * 32 + tid1] = tid1

    expect_reject(k_two_threads)


@tilelang.testing.requires_cuda
def test_scale_nested_cluster_prefix_block_thread():
    # Scope-tree: a two-level cluster prefix (cluster -> cluster -> block) lowers
    # with the cluster-rank block as block_rank_in_cluster and cluster_dims from
    # the block workgroup.
    @T.prim_func
    def kernel(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4,)) as cx:
            with T.scale("cluster", workgroup=(2,)) as cy:
                with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                    with T.scale("thread", workgroup=(32,)) as tid:
                        out[((cx * 2 + cy) * 2 + cta_id) * 8 + tid] = tid

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    assert "block_rank_in_cluster" in source
    device_func = next(iter(artifact.device_mod.functions.values()))
    assert list(device_func.attrs["cluster_dims"]) == [2, 1, 1]


@tilelang.testing.requires_cuda
def test_scale_cluster_block_double_edge_interleave():
    # Scope-tree: no-op statements on BOTH the cluster-to-block edge and the
    # block-to-thread edge in the same kernel.
    @T.prim_func
    def kernel(out: T.Tensor((256,), T.int32)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            T.assume(cm >= 0)
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                for _i in T.serial(1):
                    with T.scale("thread", workgroup=(32,)) as tid:
                        out[(cm * 2 + cta_id) * 32 + tid] = cn + tid

    _assert_cluster_block_thread_interleave(tilelang.lower(kernel, target="cuda"))


@tilelang.testing.requires_cuda
def test_scale_scope_tree_extra_rejects():
    # Scope-tree centralized rejects: multiple launch groups (sibling top-level
    # scales / sibling blocks), a scale inside an `if`, and physical-path
    # interleaving must all loud-error.
    import pytest

    def expect_reject(fn):
        with pytest.raises(Exception) as ei:
            tilelang.lower(fn, target="cuda")
        msg = str(ei.value)
        assert ("not supported yet" in msg
                and "scale" in msg), f"unexpected error: {msg[:200]}"

    # Multiple launch groups: two sibling blocks directly under the root.
    @T.prim_func
    def k_two_groups(out: T.Tensor((256,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx0:
            with T.scale("thread", workgroup=(32,)) as t0:
                out[bx0 * 32 + t0] = t0
        with T.scale("block", workgroup=(4,)) as bx1:
            with T.scale("thread", workgroup=(32,)) as t1:
                out[bx1 * 32 + t1] = t1

    expect_reject(k_two_groups)

    # block -> if -> thread: a scale inside an `if` branch.
    @T.prim_func
    def k_block_if_thread(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            if bx >= 0:
                with T.scale("thread", workgroup=(32,)) as tid:
                    out[bx * 32 + tid] = tid

    expect_reject(k_block_if_thread)

    # Physical path interleaving: a statement between die and sm-cluster is not
    # allowed (physical scales must stay strictly direct-chained).
    @T.prim_func
    def k_physical_interleave(out: T.Tensor((256,), T.int32)):
        with T.scale("die", 2) as die:
            T.assume(die >= 0)
            with T.scale("sm-cluster", 2, cluster_size=2) as cta_id:
                with T.scale("sm", 4, num_sms_per_die=8, cluster_size=2) as local_cluster:
                    with T.scale("thread", 128) as tx:
                        if tx == 0:
                            out[die] = cta_id

    expect_reject(k_physical_interleave)


@tilelang.testing.requires_cuda
def test_scale_inner_serial_loop_allowed():
    # A plain non-scale loop with NO inner scale is fine: it must lower normally,
    # not be over-rejected by the guard.
    @T.prim_func
    def kernel(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            with T.scale("thread", workgroup=(32,)) as tid:
                for i in T.serial(1):
                    if tid == 0:
                        out[bx * 32 + i] = bx

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


@tilelang.testing.requires_cuda
def test_scale_block_cluster_order_rejected():
    # The legal hierarchy is `cluster -> block -> thread`. The reversed order
    # `block -> cluster` is invalid and must raise a clear error early, not fail
    # late as an undefined-var error.
    import pytest

    @T.prim_func
    def kernel(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            with T.scale("cluster", workgroup=(2,)) as cx:
                with T.scale("thread", workgroup=(32,)) as tid:
                    if tid == 0:
                        out[bx * 32] = cx

    with pytest.raises(Exception) as ei:
        tilelang.lower(kernel, target="cuda")
    msg = str(ei.value)
    assert "cluster" in msg and "block" in msg and "not supported" in msg, \
        f"unexpected error: {msg[:200]}"


@tilelang.testing.requires_cuda
def test_scale_anchor_sibling_statements_preserved():
    # Anchor-local wrap: the launch attrs must wrap only the scale anchor, not
    # the no-op sibling statements that precede it in the PrimFunc body SeqStmt.
    # This test locks the "siblings present -> no crash / not wrongly wrapped
    # into a broken lowering" path; the scope-tree rewriter replaces only the
    # anchor For in place and leaves siblings where they are.
    #
    # Note: only *leading* siblings are exercised. A statement trailing the scale
    # scope hits a pre-existing frontend empty-SeqStmt quirk during root-block
    # construction (unrelated to LowerScaleLaunch), so it is not tested here.
    @T.prim_func
    def kernel(out: T.Tensor((128,), T.int32)):
        T.assume(True)      # no-op sibling before the scale anchor
        T.evaluate(0)       # second no-op sibling before the anchor
        with T.scale("block", workgroup=(4,)) as bx:
            with T.scale("thread", workgroup=(32,)) as tid:
                out[bx * 32 + tid] = tid

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


@tilelang.testing.requires_cuda
def test_scale_thread_body_side_effects_allowed():
    # IsNoOpPathToChild must not scan into the terminal thread's own body: a
    # block -> T.serial -> thread whose thread body contains alloc / local store
    # / if / store must lower fine (those are per-thread kernel body, not
    # between-level side effects).
    @T.prim_func
    def kernel(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            for _i in T.serial(1):
                with T.scale("thread", workgroup=(32,)) as tid:
                    acc = T.alloc_local((1,), T.int32)
                    acc[0] = bx
                    if tid < 16:
                        out[bx * 32 + tid] = acc[0]

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source


@tilelang.testing.requires_cuda
def test_scale_swizzle_with_scope_tree_interleave():
    # swizzle= must still emit the threadblock_swizzle_pattern attr when a no-op
    # statement is interleaved on the block-to-thread edge (scope-tree path).
    @T.prim_func
    def kernel(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,), swizzle=10) as bx:
            for _i in T.serial(1):
                with T.scale("thread", workgroup=(32,)) as tid:
                    out[bx * 32 + tid] = tid

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source
    assert "blockIdx.x" in source
    assert "threadIdx.x" in source
    assert "rasterization2DRow" in source


# ---------------------------------------------------------------------------
# Top-Down Scale Expansion skeleton (region parser + fail-closed planner).
#
# These exercise the generic, op-agnostic normalizer skeleton
# (tilelang/tileop/scale_expansion.py + transform/normalize_scale_expansion.py).
# The parser is read-only and the planner performs no rewrite yet (no expansion
# template is registered), so every multi-scale tile-op / interleaving segment
# must loud-error from the generic planner -- NOT be handled by a special-case
# pass. block -> T.copy -> thread therefore still rejects (see
# test_scale_block_interleave_rejects), via strict LowerScaleLaunch.
# ---------------------------------------------------------------------------


def _regionize(fn):
    import tvm
    from tilelang.transform import build_region_tree
    mod = tvm.IRModule.from_expr(fn.with_attr("global_symbol", "main"))
    return build_region_tree(mod["main"])


@requires_expansion
def test_scale_region_parser_device_gemm():
    # device -> T.gemm (direct): a single device-scope tile-op segment.
    from tilelang.tileop.scale_expansion import ScaleRegion, ScaleSegment

    @T.prim_func
    def k(A: T.Tensor((128, 128), T.float16), B: T.Tensor((128, 128), T.float16),
          C: T.Tensor((128, 128), T.float16)):
        with T.scale("device"):
            T.gemm(A, B, C)

    root = _regionize(k)
    assert root.is_root
    device = root.child_regions()
    assert len(device) == 1 and device[0].scale_name == "device"
    segs = device[0].segments()
    assert len(segs) == 1
    assert segs[0].op_name == "tl.tileop.gemm"
    assert not segs[0].is_side_effect


@requires_expansion
def test_scale_expansion_device_gemm_via_generic_path():
    # Milestone 10: a direct device -> T.gemm is now handled by the generic
    # top-down expansion path (GemmDeviceExpansionTemplate), NOT the legacy
    # PrepareScaleTileOps whole-function rewrite. Prove the dispatch: the registry
    # resolves ("device", "tl.tileop.gemm") to a ScaleExpansionTemplate, and the
    # generic planner yields a device->block replace_func plan whose replacement
    # is the generated device/block/thread kernel.
    import tvm
    from tilelang.tileop.scale_expansion import (
        resolve_scale_expansion_template,
        ensure_default_scale_expansion_templates_registered)
    from tilelang.transform import plan_region_expansions

    ensure_default_scale_expansion_templates_registered()
    tmpl = resolve_scale_expansion_template("device", "tl.tileop.gemm")
    assert tmpl is not None
    assert tmpl.from_scale == "device" and tmpl.to_scale == "block"

    @T.prim_func
    def k(A: T.Tensor((128, 128), T.float16), B: T.Tensor((128, 128), T.float16),
          C: T.Tensor((128, 128), T.float16)):
        with T.scale("device"):
            T.gemm(A, B, C)

    mod = tvm.IRModule.from_expr(k.with_attr("global_symbol", "main"))
    plans = plan_region_expansions(mod["main"])
    assert len(plans) == 1
    plan = plans[0]
    assert plan.kind == "replace_func"
    assert plan.from_scale == "device" and plan.to_scale == "block"
    assert plan.replacement_func is not None


@requires_expansion
def test_scale_expansion_device_gemm_rewritten_by_normalize_pass():
    # Pipeline-level proof of the migration: running ONLY BindTarget +
    # NormalizeScaleExpansion (no PrepareScaleTileOps) on a direct device GEMM
    # rewrites it into a device -> block -> thread tree. The device-scope GEMM is
    # gone and block/thread scale loops appear -- i.e. the generic normalizer, not
    # the legacy whole-function pass, performs the rewrite in the pipeline.
    import tvm
    import tilelang.transform as TT

    @T.prim_func
    def k(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16),
          C: T.Tensor((256, 256), T.float16)):
        with T.scale("device") as d:
            T.gemm(A, B, C)

    target = tvm.target.Target("cuda")
    mod = tvm.IRModule.from_expr(k.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = TT.NormalizeScaleExpansion()(mod)
    script = mod["main"].script()
    # device scale launch loop kept as outer scope; block + thread launch loops
    # now present (the GEMM expanded into a device -> block -> thread tree).
    assert '"tl.scale.name": "block"' in script
    assert '"tl.scale.name": "thread"' in script
    assert '"tl.scale.name": "device"' in script
    # The GEMM now lives at the innermost (thread) scope: its scale_ctx.path is
    # the full device->block->thread chain, not a bare device-scope op.
    assert 'tl.scale_ctx.path=["device", "block", "thread"]' in script
    assert "T.gemm(" in script


@requires_expansion
def test_scale_expansion_retags_moved_tileop_to_child_scale():
    # When NormalizeScaleExpansion relocates the block-scope copy into the thread
    # region, the moved tl.tileop.copy's scale_ctx metadata must be retagged to
    # the destination scale: name == "thread", path == ["block", "thread"].
    # (Generic retag in the splice stage, not a copy special case.)
    import tvm
    from tvm import tirx as tir
    from tvm.tirx.stmt_functor import post_order_visit
    import tilelang.transform as TT

    @T.prim_func
    def k_copy(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            smem = T.alloc_shared((64, 64), T.float16)
            T.copy(A[bx, 0, 0], smem)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[bx, tid] = smem[tid, 0]

    target = tvm.target.Target("cuda")
    mod = tvm.IRModule.from_expr(k_copy.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = TT.PrepareScaleTileOps()(mod)
    mod = TT.NormalizeScaleExpansion()(mod)

    found = []

    def visit(node):
        if (isinstance(node, tir.Call) and isinstance(node.op, tir.op.Op)
                and node.op.name == "tl.tileop.copy"):
            ann = dict(node.annotations) if node.annotations else {}
            name = ann.get("tl.scale_ctx.name")
            path = ann.get("tl.scale_ctx.path")
            name = name.value if hasattr(name, "value") else name
            path = [p.value if hasattr(p, "value") else p for p in (path or [])]
            found.append((name, path))

    post_order_visit(mod["main"].body, visit)
    assert len(found) == 1, f"expected exactly one moved copy, got {found}"
    name, path = found[0]
    assert name == "thread", f"copy scale_ctx.name should be retagged to thread, got {name}"
    assert path == ["block", "thread"], \
        f"copy scale_ctx.path should be [block, thread], got {path}"


@requires_expansion
def test_scale_region_parser_block_copy_thread():
    # block -> T.copy -> thread: a block-scope copy segment, then a thread child
    # region. The parser must distinguish the block copy Segment from the
    # ChildRegion(thread) and its inner store Segment.
    from tilelang.tileop.scale_expansion import ScaleRegion, ScaleSegment

    @T.prim_func
    def k(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            smem = T.alloc_shared((64, 64), T.float16)
            T.copy(A[bx, 0, 0], smem)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[bx, tid] = smem[tid, 0]

    root = _regionize(k)
    block = root.child_regions()
    assert len(block) == 1 and block[0].scale_name == "block"
    # Ordered items: a block copy Segment, then a thread ChildRegion.
    items = block[0].items
    assert isinstance(items[0], ScaleSegment)
    assert items[0].scale_name == "block"
    assert items[0].op_name == "tl.tileop.copy"
    assert isinstance(items[1], ScaleRegion)
    assert items[1].scale_name == "thread"
    # The thread child holds the leaf store as a side-effect segment.
    thread_segs = items[1].segments()
    assert len(thread_segs) == 1
    assert thread_segs[0].op_name is None
    assert thread_segs[0].is_side_effect


@requires_expansion
def test_scale_region_parser_block_store_thread():
    # block -> BufferStore -> thread: a raw side-effect segment between scales.
    from tilelang.tileop.scale_expansion import ScaleRegion, ScaleSegment

    @T.prim_func
    def k(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            out[bx * 32] = bx
            with T.scale("thread", workgroup=(32,)) as tid:
                out[bx * 32 + tid] = tid

    root = _regionize(k)
    block = root.child_regions()[0]
    seg = block.items[0]
    assert isinstance(seg, ScaleSegment)
    assert seg.op_name is None
    assert seg.is_side_effect
    assert isinstance(block.items[1], ScaleRegion)
    assert block.items[1].scale_name == "thread"


@requires_expansion
def test_scale_region_parser_cluster_block_copy_thread():
    # cluster -> block -> T.copy -> thread: the block copy segment must be
    # discoverable nested under a cluster region.
    from tilelang.tileop.scale_expansion import ScaleRegion, ScaleSegment

    @T.prim_func
    def k(A: T.Tensor((256, 256), T.float16), B: T.Tensor((256, 256), T.float16)):
        with T.scale("cluster", workgroup=(4, 2)) as (cm, cn):
            with T.scale("block", workgroup=(2, 1, 1)) as cta_id:
                smem = T.alloc_shared((64, 64), T.float16)
                T.copy(A[(cm * 2 + cta_id) * 16, 0], smem)
                with T.scale("thread", workgroup=(32,)) as tid:
                    B[(cm * 2 + cta_id) * 16 + tid, 0] = smem[0, 0]

    root = _regionize(k)
    # cluster region (the parser may model the multi-axis cluster as nested
    # serial scale loops of the same name); descend to the block region.
    def find_scale(reg, name):
        if reg.scale_name == name:
            return reg
        for c in reg.child_regions():
            r = find_scale(c, name)
            if r is not None:
                return r
        return None

    cluster = find_scale(root, "cluster")
    assert cluster is not None
    block = find_scale(root, "block")
    assert block is not None
    copy_segs = [it for it in block.items
                 if isinstance(it, ScaleSegment) and it.op_name == "tl.tileop.copy"]
    assert len(copy_segs) == 1
    thread = find_scale(block, "thread")
    assert thread is not None


@requires_expansion
def test_scale_expansion_template_resolves_block_copy():
    # The block-scope copy expansion goes through the generic
    # ScaleExpansionTemplate dispatch path, NOT a standalone special-case pass.
    # Prove it two ways: (1) the registry resolves ("block", "tl.tileop.copy");
    # (2) the generic planner produces a block->thread ExpansionPlan with a
    # sync_threads barrier-after for the k_copy region.
    import tvm
    from tilelang.tileop.scale_expansion import (
        resolve_scale_expansion_template,
        ensure_default_scale_expansion_templates_registered,
    )
    from tilelang.transform import plan_region_expansions

    ensure_default_scale_expansion_templates_registered()
    tmpl = resolve_scale_expansion_template("block", "tl.tileop.copy")
    assert tmpl is not None
    assert tmpl.from_scale == "block" and tmpl.to_scale == "thread"
    assert "tl.tileop.copy" in tmpl.op_names

    @T.prim_func
    def k(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            smem = T.alloc_shared((64, 64), T.float16)
            T.copy(A[bx, 0, 0], smem)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[bx, tid] = smem[tid, 0]

    mod = tvm.IRModule.from_expr(k.with_attr("global_symbol", "main"))
    plans = plan_region_expansions(mod["main"])
    assert len(plans) == 1
    plan = plans[0]
    assert plan.from_scale == "block" and plan.to_scale == "thread"
    assert len(plan.lowered_stmts) == 1
    # Option B: the template reports producer effects (writes_shared) and lets the
    # generic BarrierPlanner derive the CTA sync -- it does NOT hard-code a
    # sync_threads barrier in the plan.
    assert plan.effects.writes_shared is True
    from tilelang.tileop.scale_barrier_planner import (
        ScaleDependencyAnalysis, BarrierPlanner)
    from tilelang.tileop.scale_expansion import MemoryEffects
    deps = ScaleDependencyAnalysis().required_dependencies(
        "block", plan.effects, MemoryEffects(reads_shared=True))
    assert len(deps) == 1
    assert BarrierPlanner().plan_barrier(deps[0]).kind == "sync_threads"

    # (3) template.match() must accept the block-copy segment (the finer-grained
    # predicate the normalizer gates on before decode/validate/plan).
    from tilelang.transform import build_region_tree
    from tilelang.tileop.scale_expansion import ExpansionContext
    root = build_region_tree(mod["main"])
    block_region = root.child_regions()[0]
    copy_seg = block_region.segments()[0]
    ctx = ExpansionContext(region=block_region, func=mod["main"], target=None)
    assert tmpl.match(copy_seg, ctx) is True


@requires_expansion
def test_scale_expansion_template_resolves_block_fill():
    # Milestone 13: the block-scope elementwise fill expansion goes through the
    # generic ScaleExpansionTemplate dispatch path. (1) the registry resolves
    # ("block", "tl.tileop.fill"); (2) the generic planner produces a block->thread
    # plan reporting writes_shared (for a shared fill); (3) a fragment fill reports
    # writes_local so NO block sync is derived (sync is not over-introduced).
    import tvm
    from tilelang.tileop.scale_expansion import (
        resolve_scale_expansion_template,
        ensure_default_scale_expansion_templates_registered,
        MemoryEffects,
    )
    from tilelang.transform import plan_region_expansions
    from tilelang.tileop.scale_barrier_planner import (
        ScaleDependencyAnalysis, BarrierPlanner)

    ensure_default_scale_expansion_templates_registered()
    tmpl = resolve_scale_expansion_template("block", "tl.tileop.fill")
    assert tmpl is not None
    assert tmpl.from_scale == "block" and tmpl.to_scale == "thread"
    assert "tl.tileop.fill" in tmpl.op_names

    # Shared fill -> writes_shared -> a block sync is derivable.
    @T.prim_func
    def k_shared(B: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            smem = T.alloc_shared((64, 64), T.float16)
            T.fill(smem, 0)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[bx, tid] = smem[tid, 0]

    mod = tvm.IRModule.from_expr(k_shared.with_attr("global_symbol", "main"))
    plans = plan_region_expansions(mod["main"])
    assert len(plans) == 1
    plan = plans[0]
    assert plan.from_scale == "block" and plan.to_scale == "thread"
    assert plan.effects.writes_shared is True
    deps = ScaleDependencyAnalysis().required_dependencies(
        "block", plan.effects, MemoryEffects(reads_shared=True))
    assert len(deps) == 1
    assert BarrierPlanner().plan_barrier(deps[0]).kind == "sync_threads"

    # Fragment fill -> writes_local (thread-owned) -> NO block sync derived.
    @T.prim_func
    def k_frag(B: T.Tensor((4, 32), T.float32)):
        with T.scale("block", workgroup=(4,)) as bx:
            frag = T.alloc_fragment((32,), T.float32)
            T.fill(frag, 0)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[bx, tid] = frag[tid]

    mod2 = tvm.IRModule.from_expr(k_frag.with_attr("global_symbol", "main"))
    plans2 = plan_region_expansions(mod2["main"])
    assert len(plans2) == 1
    assert plans2[0].effects.writes_local is True
    assert plans2[0].effects.writes_shared is False
    deps2 = ScaleDependencyAnalysis().required_dependencies(
        "block", plans2[0].effects, MemoryEffects(reads_local=True))
    assert deps2 == [], "fragment fill must not derive a block sync"


@requires_expansion
def test_scale_expansion_planner_rejects_unregistered_edge():
    # An expandable tile-op segment on an edge with NO registered template must
    # loud-error (fail-closed). T.copy is registered for the block scale, NOT the
    # device scale, so a device-scope T.copy is an unregistered (from_scale,
    # op_name) edge.
    import tvm
    import pytest
    from tilelang.transform import plan_region_expansions

    @T.prim_func
    def k(A: T.Tensor((64, 64), T.float16), B: T.Tensor((64, 64), T.float16)):
        with T.scale("device") as d:
            smem = T.alloc_shared((64, 64), T.float16)
            T.copy(A, smem)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[tid, 0] = smem[tid, 0]

    mod = tvm.IRModule.from_expr(k.with_attr("global_symbol", "main"))
    with pytest.raises(NotImplementedError) as ei:
        plan_region_expansions(mod["main"])
    msg = str(ei.value)
    assert "no expansion template registered" in msg
    assert "device" in msg and "tl.tileop.copy" in msg


@requires_expansion
def test_scale_expansion_planner_rejects_raw_store_segment():
    # A raw BufferStore interleaved with a child scale must be rejected by the
    # generic planner (no op template can claim a side-effect segment).
    import tvm
    import pytest
    from tilelang.transform import plan_region_expansions

    @T.prim_func
    def k(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            out[bx * 32] = bx
            with T.scale("thread", workgroup=(32,)) as tid:
                out[bx * 32 + tid] = tid

    mod = tvm.IRModule.from_expr(k.with_attr("global_symbol", "main"))
    with pytest.raises(NotImplementedError) as ei:
        plan_region_expansions(mod["main"])
    assert "raw side-effect statement" in str(ei.value)


@requires_expansion
def test_scale_expansion_planner_noop_for_clean_tree():
    # A clean nested device -> block -> thread tree (tile ops only at the leaf
    # thread scale) has nothing to expand: the planner returns an empty plan list.
    import tvm
    from tilelang.transform import plan_region_expansions

    @T.prim_func
    def k(out: T.Tensor((128,), T.int32)):
        with T.scale("block", workgroup=(4,)) as bx:
            with T.scale("thread", workgroup=(32,)) as tid:
                out[bx * 32 + tid] = tid

    mod = tvm.IRModule.from_expr(k.with_attr("global_symbol", "main"))
    assert plan_region_expansions(mod["main"]) == []


# ---------------------------------------------------------------------------
# Scale-parametric analysis: ScaleSemantics + BarrierPlanner + StagePlanner.
# ---------------------------------------------------------------------------


@requires_expansion
def test_classify_storage_scope():
    # The shared scope->memory-class helper used by the copy/fill templates.
    from tilelang.tileop.scale_expansion import (
        classify_storage_scope,
        memory_effect_read_for_scope,
        memory_effect_write_for_scope,
    )

    assert classify_storage_scope("shared") == "shared"
    assert classify_storage_scope("shared.dyn") == "shared"
    assert classify_storage_scope("local") == "local"
    assert classify_storage_scope("local.fragment") == "local"
    assert classify_storage_scope("fragment") == "local"
    assert classify_storage_scope("warp") == "local"
    assert classify_storage_scope("global") == "global"
    assert classify_storage_scope("") == "global"
    assert classify_storage_scope(None) == "global"
    assert classify_storage_scope("weird") == "global"

    # The read/write effect helpers set exactly the matching flag.
    assert memory_effect_write_for_scope("shared.dyn").writes_shared is True
    assert memory_effect_write_for_scope("shared.dyn").writes_global is False
    assert memory_effect_write_for_scope("fragment").writes_local is True
    assert memory_effect_write_for_scope("global").writes_global is True
    assert memory_effect_read_for_scope("shared").reads_shared is True
    assert memory_effect_read_for_scope(None).reads_global is True


@requires_expansion
def test_scale_semantics_registry_resolves_core_scales():
    # ScaleSemantics now describes only hierarchy + sync capability (NOT memory
    # visibility). Storage orderability lives in StorageSemantics.
    from tilelang.tileop.scale_semantics import (
        resolve_scale_semantics, resolve_storage_semantics)

    thread = resolve_scale_semantics("thread")
    assert thread is not None
    assert thread.default_barrier is None
    assert thread.supports_in_kernel_barrier is False
    # ScaleSemantics must no longer carry a memory-visibility field.
    assert not hasattr(thread, "visible_memory_spaces")

    block = resolve_scale_semantics("block")
    assert block is not None
    assert block.default_barrier is not None
    assert block.default_barrier.scope == "block"
    assert block.default_barrier.kind == "sync_threads"
    assert block.supports_in_kernel_barrier is True

    device = resolve_scale_semantics("device")
    assert device is not None
    assert device.default_barrier.scope == "device"
    assert device.default_barrier.kind == "launch_boundary"
    assert device.supports_in_kernel_barrier is False
    assert device.supports_stage_boundary is True

    # cluster / node remain fail-closed (sync model not encoded).
    assert resolve_scale_semantics("cluster").sync_modeled is False
    assert resolve_scale_semantics("node").sync_modeled is False

    # StorageSemantics: shared is block-barrier-orderable; register/local/
    # fragment are program-order-only (no block_barrier mode); global is
    # launch-boundary-orderable.
    shared = resolve_storage_semantics("shared")
    assert shared is not None
    assert any(m.kind == "block_barrier" and m.executor_scale == "block"
               for m in shared.ordering_modes)

    for space in ("register", "local", "fragment"):
        st = resolve_storage_semantics(space)
        assert st is not None, space
        assert st.storage_instance_scope == "thread"
        assert all(m.kind == "program_order" for m in st.ordering_modes), space
        assert not any(m.kind == "block_barrier" for m in st.ordering_modes), space

    glob = resolve_storage_semantics("global")
    assert glob is not None
    assert any(m.kind == "launch_boundary" and m.executor_scale == "device"
               for m in glob.ordering_modes)


@requires_expansion
def test_barrier_planner_block_shared_dependency():
    # block-scope shared producer + thread shared consumer -> block sync_threads,
    # derived from the shared storage's block_barrier ordering mode.
    from tilelang.tileop.scale_expansion import MemoryEffects
    from tilelang.tileop.scale_barrier_planner import (
        ScaleDependencyAnalysis, BarrierPlanner, StagePlanner)

    deps = ScaleDependencyAnalysis().required_dependencies(
        "block",
        MemoryEffects(reads_global=True, writes_shared=True),
        MemoryEffects(reads_shared=True))
    assert len(deps) == 1 and deps[0].memory_space == "shared"
    assert deps[0].ordering_mode.kind == "block_barrier"
    spec = BarrierPlanner().plan_barrier(deps[0])
    assert spec.scope == "block" and spec.kind == "sync_threads"
    assert spec.memory_scope == "shared"
    decision = StagePlanner().plan_stage(spec)
    assert decision.in_kernel is True and decision.stage_boundary is False


@requires_expansion
def test_barrier_planner_global_at_device_launch_boundary():
    # device-scope global producer + global consumer -> launch_boundary, derived
    # from the global storage's launch_boundary ordering mode (skeleton only).
    from tilelang.tileop.scale_expansion import MemoryEffects
    from tilelang.tileop.scale_barrier_planner import (
        ScaleDependencyAnalysis, BarrierPlanner, StagePlanner)

    deps = ScaleDependencyAnalysis().required_dependencies(
        "device",
        MemoryEffects(writes_global=True),
        MemoryEffects(reads_global=True))
    assert len(deps) == 1 and deps[0].memory_space == "global"
    assert deps[0].ordering_mode.kind == "launch_boundary"
    spec = BarrierPlanner().plan_barrier(deps[0])
    assert spec.scope == "device" and spec.kind == "launch_boundary"
    decision = StagePlanner().plan_stage(spec)
    assert decision.in_kernel is False and decision.stage_boundary is True


@requires_expansion
def test_barrier_planner_local_is_not_block_orderable():
    # register/local/fragment are program-order-only: a block-scope local
    # producer + local consumer must NOT yield a block sync dependency. A block
    # barrier cannot order thread-owned register/local/fragment storage; a
    # cross-thread dependency on it must be carried by an explicit primitive.
    from tilelang.tileop.scale_expansion import MemoryEffects
    from tilelang.tileop.scale_barrier_planner import ScaleDependencyAnalysis

    deps = ScaleDependencyAnalysis().required_dependencies(
        "block",
        MemoryEffects(writes_local=True),
        MemoryEffects(reads_local=True))
    assert deps == [], f"local dependency should not be block-orderable, got {deps}"


@requires_expansion
def test_block_barrier_lowers_to_storage_sync_in_k_copy():
    # End-to-end: the block sync derived by the planner lowers to
    # tvm_storage_sync("shared") in the normalized k_copy IR (one occurrence,
    # after the relocated copy, before the thread read).
    import tvm
    from tvm import tirx as tir
    from tvm.tirx.stmt_functor import post_order_visit
    import tilelang.transform as TT

    @T.prim_func
    def k_copy(A: T.Tensor((4, 64, 64), T.float16), B: T.Tensor((4, 32), T.float16)):
        with T.scale("block", workgroup=(4,)) as bx:
            smem = T.alloc_shared((64, 64), T.float16)
            T.copy(A[bx, 0, 0], smem)
            with T.scale("thread", workgroup=(32,)) as tid:
                B[bx, tid] = smem[tid, 0]

    target = tvm.target.Target("cuda")
    mod = tvm.IRModule.from_expr(k_copy.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = TT.PrepareScaleTileOps()(mod)
    mod = TT.NormalizeScaleExpansion()(mod)

    syncs = []

    def visit(node):
        if (isinstance(node, tir.Call) and isinstance(node.op, tir.op.Op)
                and node.op.name == "tirx.tvm_storage_sync"):
            arg0 = node.args[0]
            syncs.append(arg0.value if hasattr(arg0, "value") else arg0)

    post_order_visit(mod["main"].body, visit)
    assert syncs == ["shared"], f"expected one shared storage sync, got {syncs}"


@requires_expansion
def test_device_launch_boundary_is_skeleton_only():
    # The device launch boundary is designed but NOT executable: the planner
    # produces it, but lowering it must loud-error (no multi-kernel staging yet).
    import pytest
    from tilelang.tileop.scale_expansion import MemoryEffects
    from tilelang.tileop.scale_barrier_planner import (
        ScaleDependencyAnalysis, BarrierPlanner)
    from tilelang.transform.normalize_scale_expansion import _barrier_to_stmt

    deps = ScaleDependencyAnalysis().required_dependencies(
        "device",
        MemoryEffects(writes_global=True),
        MemoryEffects(reads_global=True))
    assert len(deps) == 1
    spec = BarrierPlanner().plan_barrier(deps[0])
    assert spec.scope == "device" and spec.kind == "launch_boundary"

    with pytest.raises(NotImplementedError) as ei:
        _barrier_to_stmt(spec)
    msg = str(ei.value)
    assert "device" in msg and "not implemented yet" in msg


@requires_expansion
def test_cluster_barrier_fail_closed():
    # cluster synchronization is not modeled yet: a dependency whose ordering
    # mode executes at cluster scope must loud-error in the BarrierPlanner rather
    # than silently emit nothing. (distributed_shared carries a cluster_sync
    # ordering mode whose executor scale `cluster` has sync_modeled=False.)
    import pytest
    from tilelang.tileop.scale_semantics import resolve_storage_semantics
    from tilelang.tileop.scale_barrier_planner import (
        ScaleDependency, BarrierPlanner)

    storage = resolve_storage_semantics("distributed_shared")
    assert storage is not None
    cluster_mode = next(m for m in storage.ordering_modes
                        if m.kind == "cluster_sync")
    dep = ScaleDependency(scale_name="cluster",
                          memory_space="distributed_shared",
                          ordering_mode=cluster_mode)
    with pytest.raises(NotImplementedError) as ei:
        BarrierPlanner().plan_barrier(dep)
    assert "not modeled yet" in str(ei.value) or "not supported yet" in str(ei.value)


if __name__ == "__main__":
    tilelang.testing.main()