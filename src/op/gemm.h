/*!
 * \file tl/op/gemm.h
 * \brief Define gemm operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_H_
#define TVM_TL_OP_GEMM_H_

#include "operator.h"

namespace tvm {

namespace tl {

using namespace tir;

enum class GemmWarpPolicyType : uint8_t {
  kSquare = 0,
  kFullRow = 1,
  kFullCol = 2,
  kFree = 3,
};

// Target GEMM instruction
enum class GemmInst : uint8_t { kMMA, kWGMMA, kTCGEN5MMA, kMFMA };
class GemmWarpPolicyNode : public Object {
public:
  mutable int m_warp{0};
  mutable int n_warp{0};
  int policy_type;

  static constexpr const char *_type_key = "tl.GemmWarpPolicy";
  TVM_DECLARE_FINAL_OBJECT_INFO(GemmWarpPolicyNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmWarpPolicyNode>()
        .def_ro("policy_type", &GemmWarpPolicyNode::policy_type)
        .def_ro("m_warp", &GemmWarpPolicyNode::m_warp)
        .def_ro("n_warp", &GemmWarpPolicyNode::n_warp);
  }

  bool SEqualReduce(const GemmWarpPolicyNode *other,
                    SEqualReducer equal) const {
    return equal(policy_type, other->policy_type) &&
           equal(m_warp, other->m_warp) && equal(n_warp, other->n_warp);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(policy_type);
    hash_reduce(m_warp);
    hash_reduce(n_warp);
  }

  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;

  std::pair<int, int> ComputeWarpPartition(int M, int N, int block_size,
                                           Target target,
                                           GemmInst gemm_inst) const;

  bool isSquare() const {
    return policy_type == int(GemmWarpPolicyType::kSquare);
  }
  bool isFullRow() const {
    return policy_type == int(GemmWarpPolicyType::kFullRow);
  }
  bool isFullCol() const {
    return policy_type == int(GemmWarpPolicyType::kFullCol);
  }
  bool isFree() const { return policy_type == int(GemmWarpPolicyType::kFree); }
};

class GemmWarpPolicy : public ObjectRef {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(GemmWarpPolicy, ObjectRef, GemmWarpPolicyNode);

  explicit GemmWarpPolicy(GemmWarpPolicyType policy_type) {
    auto node = make_object<GemmWarpPolicyNode>();
    node->policy_type = (int)policy_type;
    data_ = std::move(node);
  }

  explicit GemmWarpPolicy(int policy_type) {
    auto node = make_object<GemmWarpPolicyNode>();
    node->policy_type = policy_type;
    data_ = std::move(node);
  }

  explicit GemmWarpPolicy(int m_warp, int n_warp) {
    auto node = make_object<GemmWarpPolicyNode>();
    node->m_warp = m_warp;
    node->n_warp = n_warp;
    node->policy_type = (int)GemmWarpPolicyType::kFree;
    data_ = std::move(node);
  }
};

class GemmNode : public TileOperatorNode {
public:
  bool CheckWGMMA() const;
  tir::Buffer A, B, C;
  // pointer to the A, B, C
  PrimExpr Aptr, Bptr, Cptr;
  bool trans_A, trans_B;
  int M, N, K;
  int stride_A, stride_B;
  int offset_A, offset_B;
  PrimExpr clear_accum = const_false();
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack = 1;
  int wg_wait = 0;
  PrimExpr mbarptr;
  std::optional<tir::Buffer> mbar; // mbar is optional, only used for TCGEN5MMA
  Array<PrimExpr> C_coords;
  mutable GemmWarpPolicy policy;

  static constexpr const char *_type_key = "tl.Gemm";
  TVM_DECLARE_FINAL_OBJECT_INFO(GemmNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmNode>()
        .def_ro("A", &GemmNode::A)
        .def_ro("B", &GemmNode::B)
        .def_ro("C", &GemmNode::C)
        .def_ro("Aptr", &GemmNode::Aptr)
        .def_ro("Bptr", &GemmNode::Bptr)
        .def_ro("Cptr", &GemmNode::Cptr)
        .def_ro("trans_A", &GemmNode::trans_A)
        .def_ro("trans_B", &GemmNode::trans_B)
        .def_ro("M", &GemmNode::M)
        .def_ro("N", &GemmNode::N)
        .def_ro("K", &GemmNode::K)
        .def_ro("stride_A", &GemmNode::stride_A)
        .def_ro("stride_B", &GemmNode::stride_B)
        .def_ro("offset_A", &GemmNode::offset_A)
        .def_ro("offset_B", &GemmNode::offset_B)
        .def_ro("clear_accum", &GemmNode::clear_accum)
        .def_ro("kPack", &GemmNode::kPack)
        .def_ro("wg_wait", &GemmNode::wg_wait)
        .def_ro("policy", &GemmNode::policy);
  }

  bool SEqualReduce(const GemmNode *other, SEqualReducer equal) const {
    return equal(A, other->A) && equal(B, other->B) && equal(C, other->C) &&
           equal(Aptr, other->Aptr) && equal(Bptr, other->Bptr) &&
           equal(Cptr, other->Cptr) && equal(trans_A, other->trans_A) &&
           equal(trans_B, other->trans_B) && equal(M, other->M) &&
           equal(N, other->N) && equal(K, other->K) &&
           equal(stride_A, other->stride_A) &&
           equal(stride_B, other->stride_B) &&
           equal(offset_A, other->offset_A) &&
           equal(offset_B, other->offset_B) &&
           equal(clear_accum, other->clear_accum) &&
           equal(kPack, other->kPack) && equal(wg_wait, other->wg_wait) &&
           equal(policy, other->policy);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(A);
    hash_reduce(B);
    hash_reduce(C);
    hash_reduce(Aptr);
    hash_reduce(Bptr);
    hash_reduce(Cptr);
    hash_reduce(trans_A);
    hash_reduce(trans_B);
    hash_reduce(M);
    hash_reduce(N);
    hash_reduce(K);
    hash_reduce(stride_A);
    hash_reduce(stride_B);
    hash_reduce(offset_A);
    hash_reduce(offset_B);
    hash_reduce(clear_accum);
    hash_reduce(kPack);
    hash_reduce(wg_wait);
    hash_reduce(policy);
  }
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  TileOperator Clone() const;

private:
  GemmInst GetGemmInst(int block_size, Target target) const;
  bool AllowTCGEN5MMA(Target target) const;
  bool AllowWGMMA(int block_size, Target target) const;

  mutable bool completed_ = false;
};

class Gemm : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(Gemm, TileOperator, GemmNode);
  TVM_DLL Gemm(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_H_