/*!
 * \file tl/op/gemm.h
 * \brief Define gemm operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_H_
#define TVM_TL_OP_GEMM_H_

#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class Gemm : public Operator {
public:
  Gemm(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final;
  static const Op &Get();
  enum class GemmWarpPolicy {
    kSquare = 0,
    kFullRow = 1,
    kFullCol = 2,
  } policy;

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<Gemm>(*this);
  }

private:
  // Target GEMM instruction
  enum class GemmInst { kMMA, kWGMMA, kUTCMMA, kMFMA };
  GemmInst GetGemmInst(int block_size, Target target) const;

  std::pair<int, int> ComputeWarpPartition(int num_warps, GemmInst gemm_inst,
                                           Target target) const;

  bool CheckWGMMA() const;
  Array<PrimExpr> call_args;
  tir::Buffer A, B, C;
  // pointer to the A, B, C
  PrimExpr Aptr, Bptr, Cptr;
  bool trans_A, trans_B;
  int M, N, K;
  int stride_A, stride_B;
  int offset_A, offset_B;
  bool clear_accum = false;
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack = 1;
  int wg_wait = 0;
  bool completed_ = false;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_H_