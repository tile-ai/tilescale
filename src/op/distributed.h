/*!
 * \file tl/op/distributed.h
 * \brief Distributed intrinsics.
 *
 */

#include "op.h"
#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

/*!
 * \brief tvm intrinsics for getting the PE id
 *
 * int GetPE()
 *
 */
const Op &GetPE();

/*!
 * \brief tvm intrinsics for getting the total number of PEs
 */
const Op &GetPENum();

/*!
 * \brief tvm intrinsics for getting the PE id
 *
 * int IntPE()
 *
 */
const Op &IntPE();

/*!
 * \brief tvm intrinsics for global barrier synchronization
 */
const Op &BarrierAll();

/*!
 * \brief tvm intrinsics for global barrier synchronization in a team
 */
const Op &Barrier();

/*!
 * \brief tvm intrinsics for block-level barrier synchronization
 */
const Op &BarrierAllBlock();

/*!
 * \brief tvm intrinsics for block-level barrier synchronization in a team
 */
const Op &BarrierBlock();

/*!
 * \brief tvm intrinsics for warp-level barrier synchronization
 */
const Op &BarrierAllWarp();

/*!
 * \brief tvm intrinsics for warp-level barrier synchronization in a team
 */
const Op &BarrierWarp();


/*!
 * \brief tvm intrinsics for global synchronization
 */
const Op &SyncAll();

/*!
 * \brief tvm intrinsics for global synchronization in a team
 */
const Op &Sync();

/*!
 * \brief tvm intrinsics for block-level synchronization
 */
const Op &SyncAllBlock();

/*!
 * \brief tvm intrinsics for block-level synchronization in a team
 */
const Op &SyncBlock();

/*!
 * \brief tvm intrinsics for warp-level synchronization
 */
const Op &SyncAllWarp();

/*!
 * \brief tvm intrinsics for warp-level synchronization in 
 */
const Op &SyncWarp();

/*!
 * \brief tvm intrinsics for quiet operation
 */
const Op &Quiet();

/*!
 * \brief tvm intrinsics for memory fence operation
 */
const Op &Fence();

/*!
 * \brief tvm intrinsics for non-blocking block-level memory get
 */
const Op &GetmemNbiBlock();

/*!
 * \brief tvm intrinsics for blocking block-level memory get
 */
const Op &GetmemBlock();

/*!
 * \brief tvm intrinsics for non-blocking warp-level memory get
 */
const Op &GetmemNbiWarp();

/*!
 * \brief tvm intrinsics for blocking warp-level memory get
 */
const Op &GetmemWarp();

/*!
 * \brief tvm intrinsics for non-blocking memory get
 */
const Op &GetmemNbi();

/*!
 * \brief tvm intrinsics for blocking memory get
 */
const Op &Getmem();

/*!
 * \brief tvm intrinsics for block-level memory put
 */
const Op &PutmemBlock();

/*!
 * \brief tvm intrinsics for non-blocking block-level memory put
 */
const Op &PutmemNbiBlock();

/*!
 * \brief tvm intrinsics for warp-level memory put
 */
const Op &PutmemWarp();

/*!
 * \brief tvm intrinsics for non-blocking warp-level memory put
 */
const Op &PutmemNbiWarp();

/*!
 * \brief tvm intrinsics for memory put
 */
const Op &Putmem();

/*!
 * \brief tvm intrinsics for non-blocking memory put
 */
const Op &PutmemNbi();

/*!
 * \brief tvm intrinsics for signaled memory put
 */
const Op &PutmemSignal();

/*!
 * \brief tvm intrinsics for non-blocking signaled memory put
 */
const Op &PutmemSignalNbi();

/*!
 * \brief tvm intrinsics for block-level signaled memory put
 */
const Op &PutmemSignalBlock();

/*!
 * \brief tvm intrinsics for non-blocking block-level signaled memory put
 */
const Op &PutmemSignalNbiBlock();

/*!
 * \brief tvm intrinsics for warp-level signaled memory put
 */
const Op &PutmemSignalWarp();

/*!
 * \brief tvm intrinsics for non-blocking warp-level signaled memory put
 */
const Op &PutmemSignalNbiWarp();

/*!
 * \brief tvm intrinsics for signal operation
 */
const Op &SignalOp();

/*!
 * \brief tvm intrinsics for waiting on signal
 */
const Op &SignalWaitUntil();

/*!
 * \brief tvm intrinsics for broadcast operation
 */
const Op &Broadcast();

/*!
 * \brief tvm intrinsics for warp-level broadcast
 */
const Op &BroadcastWarp();

/*!
 * \brief tvm intrinsics for block-level broadcast
 */
const Op &BroadcastBlock();

/*!
 * \brief tvm intrinsics for block-level memory broadcast
 */
const Op &BroadcastmemBlock();

/*!
 * \brief tvm intrinsics for collective gather operation
 */
const Op &Fcollect();

/*!
 * \brief tvm intrinsics for warp-level collective gather
 */
const Op &FcollectWarp();

/*!
 * \brief tvm intrinsics for block-level collective gather
 */
const Op &FcollectBlock();

/*!
 * \brief tvm intrinsics for collective gather operation
 */
const Op &CpengineCpAsync();

} // namespace tl
} // namespace tvm
