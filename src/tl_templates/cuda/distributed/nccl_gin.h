#pragma once

// Inter-node data movement via the NCCL GPU-Initiated Networking (GIN) device
// API.
//
// GIN is a device-side one-sided put/signal interface: a thread on one GPU
// writes into a peer GPU's registered memory without host involvement and
// without that memory being mapped into the local address space. This is the
// mechanism TileScale uses for peers that CUDA IPC/VMM cannot reach, i.e. peers
// on a different node.
//
// Addressing model, and how it differs from the intra-node path
// -------------------------------------------------------------
// The intra-node path in distributed.h addresses a peer as
//   peer_base + (local_ptr - local_base)
// which requires the peer's allocation to be mapped locally. GIN instead
// addresses memory as an (ncclWindow_t, offset) pair, where the window is a
// registration of a host-allocated buffer created by ncclCommWindowRegister.
// The window handle is symmetric across the communicator, so the same handle
// plus a byte offset names the corresponding bytes on every rank.
//
// TileScale registers its symmetric allocator arena as one window, so an
// existing local pointer converts to a GIN address by subtracting the arena
// base. That keeps the symmetric-arena invariant already relied on by the
// intra-node path and by remote TMA descriptor encoding.
//
// Requirements
// ------------
// NCCL >= 2.28.7 for the Device API used here (ncclDevCommCreate, nccl_device
// headers). The device comm must be created with ginForceEnable and nonzero
// ginSignalCount/ginCounterCount, and the buffers involved must live in
// registered windows.
//
// This header is only included in generated code when NCCL GIN support was
// detected at build time; see TL_ENABLE_NCCL_GIN.

// distributed.h defines the __constant__ meta_data table this header reads, and
// pulls in meta_layout.h for the offset names.
#include "distributed.h"

#if defined(TL_ENABLE_NCCL_GIN)

#include <nccl.h>
#include <nccl_device.h>

#if !defined(NCCL_VERSION_CODE) || (NCCL_VERSION_CODE < 22807)
#error "TL_ENABLE_NCCL_GIN requires NCCL >= 2.28.7 for the Device API"
#endif

namespace tl {
namespace gin {

// The device comm is passed to kernels by value in a __grid_constant__ param by
// NCCL's own examples; TileScale instead publishes the handle through the
// distributed metadata table so existing kernel signatures are unchanged.
// meta_data holds a pointer to a device-resident ncclDevComm.
TL_DEVICE ncclDevComm const *dev_comm() {
  return reinterpret_cast<ncclDevComm const *>(
      meta_data[TL_META_GLOBAL_DEV_COMM]);
}

TL_DEVICE bool available() { return meta_data[TL_META_GLOBAL_DEV_COMM] != 0; }

// The arena window registered for GIN addressing. Zero when GIN is unavailable
// or the backend is cudaMalloc (which cannot be registered as an NCCL window).
// ncclWindow_t is `struct ncclWindow_vidmem *`, so the handle stored as an
// integer in the table has to be reinterpreted, not static_cast.
TL_DEVICE ncclWindow_t arena_window() {
  return reinterpret_cast<ncclWindow_t>(meta_data[TL_META_ARENA_WINDOW]);
}

// Arena base address; subtract from a local pointer to get a GIN window offset.
TL_DEVICE uint64_t arena_base() { return meta_data[TL_META_ARENA_BASE]; }

// Convert a local arena pointer to the GIN offset valid on every rank.
// The arena is symmetric, so the same offset names the corresponding bytes on
// any peer — the identical subtraction the intra-node peer-pointer path already
// performs, just paired with a window handle instead of a peer base.
TL_DEVICE size_t arena_offset(void const *ptr) {
  return reinterpret_cast<uint64_t>(ptr) - arena_base();
}

// One GIN context is one network channel with its own QP per peer, so pinning
// every CTA to context 0 serializes the whole grid onto one QP and leaves the
// node's other NICs idle. Spreading CTAs across contexts is what would turn one
// channel's bandwidth into the fabric's -- and is the main throughput headroom
// left, since a single QP measures ~23 GB/s where one NIC's line rate is 50.
//
// Signal state is per context, and that is what makes spreading work at all.
// A put issued on sender context i increments the receiver's signal *through
// context i*, so a CTA sitting on one of C contexts observes only 1/C of the
// rank's arrivals. `wait_signal` therefore divides the caller's grid-wide target
// by context_span(); see the comment there.
//
// Measured 2026-08-03, allgather, 64 MB shard, bf16, two ranks:
//   contexts   1 -> 24.3 GB/s     2 -> 44.0 GB/s     4 -> 44.5 GB/s
// So a single QP is the throughput wall and two contexts nearly double it,
// saturating by two on this fabric. Throughput is flat in chunk count from 1 to
// 4, confirming the QP rather than message size is the limit.
//
// This took three wrong turns worth recording, because each looked convincing:
//   1. Rotating on `ginContextCount` from ncclDevCommRequirements. That field is
//      documented as only a hint -- "the actual context count in the devcomm may
//      not match" -- and the request here (8) is NOT what gets granted (4).
//      Indexing past the granted count hangs.
//   2. Blaming the kernel cache. Real hazard -- the cache key hashes the script,
//      args, target and compile flags but not this header, so editing it leaves
//      stale binaries behind -- but not this bug; TL_GIN_CONTEXTS is a -D flag
//      and so is in the key.
//   3. "Disproving" per-context signals from a 332 GB/s reading. The reading was
//      real and impossible -- above the ~63 GB/s PCIe Gen5 x16 cap on one GPU's
//      egress -- but the cause was dividing by the *requested* 8 while the device
//      had clamped to 4, so the wait was 2x too weak and the kernel returned
//      early. The conclusion drawn from it was wrong.
//
// The lesson each time: get the number off the device (TL_GIN_DEBUG=1) instead of
// inferring it, and sanity-check any bandwidth against `nvidia-smi topo -m`
// before believing it. An under-target is silent -- the host's correctness check
// syncs and runs a reference collective first, which gives in-flight RDMA writes
// time to land before the comparison reads the buffer.
//
// TL_GIN_CONTEXTS is a -D compile flag, so it is part of the kernel cache key --
// which also stops a stale entry from silently answering for a different policy.
//   0, 1  pin to context 0
//   n > 1 spread CTAs over min(n, granted) contexts
#ifndef TL_GIN_CONTEXTS
#define TL_GIN_CONTEXTS 0
#endif

// Splits the context choice between issuing and waiting, to tell apart the two
// candidate reasons a multi-context run hangs: puts on a non-zero context never
// delivering, versus a wait on a non-zero context never observing a signal that
// did arrive. With TL_GIN_WAIT_CTX0=1 the puts spread over contexts while every
// wait stays on context 0. If that passes on the honest target, signals are
// communicator-wide and only the waiting side is context-sensitive.
#ifndef TL_GIN_WAIT_CTX0
#define TL_GIN_WAIT_CTX0 0
#endif

// TL_GIN_DEBUG=1 prints what the devcomm actually granted. Direct evidence beats
// inference here: every theory about why multiple contexts hang has hinged on how
// many contexts exist and which one a CTA ends up on, and nothing else exposes
// those. One line per CTA, from thread 0 only.
#ifndef TL_GIN_DEBUG
#define TL_GIN_DEBUG 0
#endif

// How many contexts this kernel actually spreads over: the requested count
// clamped to what the devcomm granted. Measured granted = 4 on these nodes even
// though the allocator asks for 8, which is why the clamp matters -- and why the
// host must never assume its own request is the number in play.
TL_DEVICE uint32_t context_span() {
  ncclGin probe(*dev_comm(), 0);
  uint32_t want = TL_GIN_CONTEXTS < 1 ? 1u : static_cast<uint32_t>(TL_GIN_CONTEXTS);
  uint32_t const granted = probe.nContexts;
  if (granted < want) {
    want = granted;
  }
  return want < 1 ? 1u : want;
}

TL_DEVICE ncclGin make_gin() {
  uint32_t const want = context_span();
  uint32_t const use = static_cast<uint32_t>(blockIdx.x) % want;
#if TL_GIN_DEBUG
  if (threadIdx.x == 0) {
    printf("[gin] block=%u want=%u use=%u\n", blockIdx.x, want, use);
  }
#endif
  return ncclGin(*dev_comm(), static_cast<int>(use));
}

// The gin used for waits. See TL_GIN_WAIT_CTX0.
TL_DEVICE ncclGin make_gin_for_wait() {
#if TL_GIN_WAIT_CTX0
  return ncclGin(*dev_comm(), 0);
#else
  return make_gin();
#endif
}

// A put whose completion increments `signal` on the destination rank. The signal
// becomes visible to the peer only after this put's payload, and the payloads of
// preceding puts to that peer on this context, have settled -- so a peer that
// waits on the signal observes the data.
//
// dst_offset/src_offset are byte offsets into the registered windows. `peer` is
// a rank within `team`.
template <typename Coop>
TL_DEVICE void put_signal(Coop coop, ncclTeam team, int peer,
                          ncclWindow_t dst_window, size_t dst_offset,
                          ncclWindow_t src_window, size_t src_offset,
                          size_t bytes, ncclGinSignal_t signal) {
  make_gin()
      .put(team, peer, dst_window, dst_offset, src_window, src_offset, bytes,
           ncclGin_SignalInc{signal}, ncclGin_None{}, coop);
}

// A put with no remote notification. Pair with a later signal() or a barrier
// when the peer needs to know the data arrived.
template <typename Coop>
TL_DEVICE void put(Coop coop, ncclTeam team, int peer, ncclWindow_t dst_window,
                   size_t dst_offset, ncclWindow_t src_window,
                   size_t src_offset, size_t bytes) {
  make_gin()
      .put(team, peer, dst_window, dst_offset, src_window, src_offset, bytes,
           ncclGin_None{}, ncclGin_None{}, coop);
}

// Increment `signal` on `peer` without moving payload. Ordered after this
// context's preceding puts to that peer.
template <typename Coop>
TL_DEVICE void signal_peer(Coop coop, ncclTeam team, int peer,
                           ncclGinSignal_t signal) {
  make_gin()
      .signal(team, peer, ncclGin_SignalInc{signal}, coop);
}

// Block until `signal` has been incremented at least `least` times in total.
// Signals are cumulative and compared with rolling arithmetic, so callers track
// an expected running total rather than resetting between phases.
//
// `least` is the GRID-WIDE arrival count -- what the whole rank receives per
// launch. Signal state is per context, though: a put issued on sender context i
// increments the receiver's signal through context i, so a CTA sitting on one of
// `context_span()` contexts only ever observes its own share. Scaling here rather
// than on the host keeps the honest number in the caller and puts the division
// where the granted context count is actually known.
//
// Getting this wrong is silent in one direction: too small a target lets the wait
// return before the payload lands, and the host's correctness check still passes
// because it syncs and runs a reference collective first, which gives the
// in-flight writes time to arrive. The tell is bandwidth above what the hardware
// can carry -- 332 GB/s appeared this way, against a ~63 GB/s PCIe Gen5 x16 cap.
template <typename Coop>
TL_DEVICE void wait_signal(Coop coop, ncclGinSignal_t signal, uint64_t least) {
  uint64_t const span = static_cast<uint64_t>(context_span());
  make_gin_for_wait().waitSignal(coop, signal, least / span);
}

// Wait for one more increment than last observed, advancing the signal's shadow.
// This is the convenient form for a consumer draining a stream of arrivals.
template <typename Coop>
TL_DEVICE void wait_signal_next(Coop coop, ncclGinSignal_t signal) {
  make_gin_for_wait()
      .waitSignalMeetShadow(coop, signal);
}

// Make source buffers from this coop's puts safe to overwrite. This does NOT
// imply the data has landed remotely; use a signal for that.
template <typename Coop> TL_DEVICE void flush(Coop coop) {
  make_gin().flush(coop);
}

// Reset a signal to zero along with its shadow. Must not race with concurrent
// increments to the same signal.
TL_DEVICE void reset_signal(ncclGinSignal_t signal) {
  make_gin().resetSignal(signal);
}

// The communicator-wide team, whose ranks are global ranks.
TL_DEVICE ncclTeam world_team() { return ncclTeamWorld(*dev_comm()); }

// Convenience wrappers for whole-CTA and single-thread issue, which are the two
// shapes generated code uses today.
TL_DEVICE void put_signal_cta(int peer, ncclWindow_t dst_window,
                              size_t dst_offset, ncclWindow_t src_window,
                              size_t src_offset, size_t bytes,
                              ncclGinSignal_t signal) {
  put_signal(ncclCoopCta(), world_team(), peer, dst_window, dst_offset,
             src_window, src_offset, bytes, signal);
}

TL_DEVICE void wait_signal_cta(ncclGinSignal_t signal, uint64_t least) {
  wait_signal(ncclCoopCta(), signal, least);
}

// ---------------------------------------------------------------------------
// Entry points used by generated code.
//
// These take the coop as a *template* parameter and the buffers as ordinary
// pointers, which is what the lowering in src/op/nccl_gin.cc can express: it
// emits a call_extern whose callee is a string, so the scope has to be baked
// into the name rather than passed as a constructed object, and it has no way to
// name an ncclWindow_t or compute a window offset at compile time.
//
// The window and the base used for the offsets are both read from the metadata
// table here, on the device. Both pointers must be inside the allocator arena --
// a shared-memory or fragment buffer has no window and would produce a wild
// offset rather than an error.
// ---------------------------------------------------------------------------

template <typename Coop>
TL_DEVICE void put_addr(int peer, void *dst, void const *src, size_t bytes) {
  ncclWindow_t window = arena_window();
  put(Coop(), world_team(), peer, window, arena_offset(dst), window,
      arena_offset(src), bytes);
}

template <typename Coop>
TL_DEVICE void put_signal_addr(int peer, void *dst, void const *src,
                               size_t bytes, int signal) {
  ncclWindow_t window = arena_window();
  put_signal(Coop(), world_team(), peer, window, arena_offset(dst), window,
             arena_offset(src), bytes,
             static_cast<ncclGinSignal_t>(signal));
}

template <typename Coop>
TL_DEVICE void signal_peer(int peer, int signal) {
  signal_peer(Coop(), world_team(), peer,
              static_cast<ncclGinSignal_t>(signal));
}

template <typename Coop>
TL_DEVICE void wait_signal(int signal, uint64_t least) {
  wait_signal(Coop(), static_cast<ncclGinSignal_t>(signal), least);
}

template <typename Coop> TL_DEVICE void flush() { flush(Coop()); }

} // namespace gin
} // namespace tl

#endif // TL_ENABLE_NCCL_GIN
