/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_CUDA_H_
#define TVM_TL_TARGET_CODEGEN_CUDA_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <stdlib.h>
#include <string>
#include <unordered_map>

#include "target/source/codegen_c.h"

/*
 * Utility function for judging whether distributed mode is enabled.
 * This is used to determine whether to include distributed.h in the generated
 * code.
 */
static inline bool use_distributed() {
  const char *env = std::getenv("TILELANG_USE_DISTRIBUTED");
  if (env) {
    return std::string(env) == "1";
  }
  return false;
}

namespace tvm {
namespace codegen {

class CodeGenTileLangCUDA final : public CodeGenC {
public:
  CodeGenTileLangCUDA();
  std::string Finish();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f);
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final; // NOLINT(*)
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) final;      // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  void BindThreadIndex(const IterVar &iv) final; // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, DataType from,
                         DataType target) final;
  // overload visitor
  void VisitExpr_(const RampNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitStmt_(const EvaluateNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);
  void PrintFunctionSignature(const String &function_name, const PrimFunc &func,
                              std::ostream &os);

protected:
  virtual std::string GetBufferRef(DataType t, const BufferNode *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, String global_symbol,
                       const Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *op,
                           std::ostream &os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangCUDA *p);

  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;
  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable bf16
  bool enable_bf16_{false};
  // whether enable fp8
  bool enable_fp8_{false};
  // whether enable sparse gemm
  bool enable_sparse_gemm_{false};
  // whether enable int8
  bool enable_int8_{false};
  // whether enable warp shuffle intrinsics
  bool enable_warp_shuffle_{false};
  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};
  // whether need cooperative_groups.h
  bool need_cooperative_groups_{false};
  // whether need distributed.h
  bool use_distributed_{use_distributed()};
  // Op attribute map
  OpAttrMap<bool> op_need_warp_shuffle_ =
      Op::GetAttrMap<bool>("cuda.need_warp_shuffle");

  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;

  std::unordered_map<const VarNode *, std::string> fragment_shapes;
  std::unordered_map<const VarNode *, std::string> fragment_layouts;
  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangCUDA *p);
  void PrintWmmaScope(const std::string &scope, DataType t,
                      const VarNode *variable, std::ostream &os);
  int32_t GetWmmaFragmentSize(const std::string &scope, const VarNode *variable,
                              int32_t size);
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_CUDA_H_
