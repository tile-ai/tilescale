// TileScale pass

/*!
 * \file declare_symm_buffer.cc
 * \brief Declare the symmetry buffer to prepare for operators that need buffers
 * on peer's symm heap
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

#include "../op/copy.h"
#include "../op/distributed.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::transform::PassContext;

static int name_suffix_id =
    0; // Avoid name collision for symm buffers, start from 0

/* Create a PrimExpr to calculate the symmetry pointer given a local ptr and
 * target PE */
PrimExpr CalculateSymmPtr(PrimExpr ptr, PrimExpr pe) {
  PrimExpr local_rank = Call(DataType::Int(64), tl::get_rank(), {});
  PrimExpr local_base_ptr = Call(DataType::Handle(), tl::get_local_base(), {});
  PrimExpr offset_to_base =
      Sub(Call(DataType::Handle(), tl::get_uintptr_t(), {ptr}), local_base_ptr);
  PrimExpr result = Call(DataType::Handle(), tl::get_remote_base_ptr(), {pe}) +
                    offset_to_base;
  return result;
}

/*!
 * \brief Declare the symmetry buffer to prepare for operators
 * that need buffers on peer's symm heap
 */
class SymmBufferDeclarer : public StmtExprMutator {
public:
  static PrimFunc Apply(PrimFunc f) {
    if (!f->body.defined()) {
      return f;
    }

    SymmBufferDeclarer declarer;

    // Extract symm buffer info and replace them
    // The LetStmt insertion will happen in VisitStmt_ before each copy
    f.CopyOnWrite()->body = declarer.VisitStmt(f->body);

    return f;
  };

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    // Check if this Evaluate contains a Call
    if (const CallNode *call_op = op->value.as<CallNode>()) {
      auto parsed_op =
          ParseOperator(GetRef<Call>(call_op), buffer_data_to_buffer_);
      if (parsed_op.defined() && parsed_op.as<CopyNode>()) {
        // LOG(INFO) << "Found copy";

        if (parsed_op.as<CopyNode>()->is_remote_push()) {
          // LOG(INFO) << "Found remote push";

          Buffer dst = parsed_op.as<CopyNode>()->dst;
          Array<Range> dst_range = parsed_op.as<CopyNode>()->dst_range;

          // 1. Calculate symm dst ptr
          PrimExpr symm_dst_ptr_expr =
              CalculateSymmPtr(dst->data, parsed_op.as<CopyNode>()->dst_pe);
          // LOG(INFO) << "Symm dst ptr expr: " << symm_dst_ptr_expr;

          // 2. Create a let binding
          String storage_scope =
              dst->data->type_annotation.as<PointerTypeNode>()->storage_scope;
          Var symm_dst_var =
              Var(dst->name + "_symm_" + std::to_string(name_suffix_id++),
                  PointerType(PrimType(dst->dtype), storage_scope));

          // 3. Create modified dst buffer with symm var
          dst.CopyOnWrite()->data = symm_dst_var;

          // 4. Rebuild the destination region call with the modified buffer
          // RegionOp args: [BufferLoad(min_indices), access_mask, extent_0,
          // extent_1, ...]
          Array<PrimExpr> dst_region_mins;
          Array<PrimExpr> dst_region_extents;
          for (const Range &r : dst_range) {
            dst_region_mins.push_back(r->min);
            dst_region_extents.push_back(r->extent);
          }
          BufferLoad dst_load(dst, dst_region_mins);

          Array<PrimExpr> dst_region_args;
          dst_region_args.push_back(dst_load);
          dst_region_args.push_back(
              IntImm(DataType::Int(32), call_op->args[1]
                                            .as<CallNode>()
                                            ->args[1]
                                            .as<IntImmNode>()
                                            ->value)); // access_mask
          for (const PrimExpr &extent : dst_region_extents) {
            dst_region_args.push_back(extent);
          }

          // Create new Call for the destination region
          Call dst_region_call =
              Call(call_op->args[1].as<CallNode>()->dtype,
                   call_op->args[1].as<CallNode>()->op, dst_region_args,
                   call_op->args[1].as<CallNode>()->span);

          // 5. Rebuild the Copy call with modified args
          Array<PrimExpr> new_copy_args;
          new_copy_args.push_back(call_op->args[0]); // src region (unchanged)
          new_copy_args.push_back(dst_region_call);  // modified dst region
          // Copy remaining args
          for (size_t i = 2; i < call_op->args.size(); i++) {
            new_copy_args.push_back(call_op->args[i]);
          }

          // Create the modified copy call
          Call new_copy_call =
              Call(call_op->dtype, call_op->op, new_copy_args, call_op->span);

          // Wrap it in an Evaluate statement
          Stmt modified_stmt = Evaluate(new_copy_call);

          // Wrap with LetStmt that defines the symm pointer
          return LetStmt(symm_dst_var, symm_dst_ptr_expr, modified_stmt);
        } else if (parsed_op.as<CopyNode>()->is_remote_pull()) {
          // LOG(INFO) << "Found remote pull";

          Buffer src = parsed_op.as<CopyNode>()->src;
          Array<Range> src_range = parsed_op.as<CopyNode>()->src_range;

          // 1. Calculate symm src ptr
          PrimExpr symm_src_ptr_expr =
              CalculateSymmPtr(src->data, parsed_op.as<CopyNode>()->src_pe);
          // LOG(INFO) << "Symm src ptr expr: " << symm_src_ptr_expr;

          // 2. Create a let binding
          String storage_scope =
              src->data->type_annotation.as<PointerTypeNode>()->storage_scope;
          Var symm_src_var =
              Var(src->name + "_symm_" + std::to_string(name_suffix_id++),
                  PointerType(PrimType(src->dtype), storage_scope));

          // 3. Create modified src buffer with symm var
          src.CopyOnWrite()->data = symm_src_var;

          // 4. Rebuild the source region call with the modified buffer
          // RegionOp args: [BufferLoad(min_indices), access_mask, extent_0,
          // extent_1, ...]
          Array<PrimExpr> src_region_mins;
          Array<PrimExpr> src_region_extents;
          for (const Range &r : src_range) {
            src_region_mins.push_back(r->min);
            src_region_extents.push_back(r->extent);
          }
          BufferLoad src_load(src, src_region_mins);

          Array<PrimExpr> src_region_args;
          src_region_args.push_back(src_load);
          src_region_args.push_back(
              IntImm(DataType::Int(32), call_op->args[1]
                                            .as<CallNode>()
                                            ->args[1]
                                            .as<IntImmNode>()
                                            ->value)); // access_mask
          for (const PrimExpr &extent : src_region_extents) {
            src_region_args.push_back(extent);
          }

          // Create new Call for the source region
          Call src_region_call =
              Call(call_op->args[0].as<CallNode>()->dtype,
                   call_op->args[0].as<CallNode>()->op, src_region_args,
                   call_op->args[0].as<CallNode>()->span);

          // 5. Rebuild the Copy call with modified args
          Array<PrimExpr> new_copy_args;
          new_copy_args.push_back(src_region_call);  // modified src region
          new_copy_args.push_back(call_op->args[1]); // dst region (unchanged)
          // Copy remaining args
          for (size_t i = 2; i < call_op->args.size(); i++) {
            new_copy_args.push_back(call_op->args[i]);
          }

          // Create the modified copy call
          Call new_copy_call =
              Call(call_op->dtype, call_op->op, new_copy_args, call_op->span);

          // Wrap it in an Evaluate statement
          Stmt modified_stmt = Evaluate(new_copy_call);

          // Wrap with LetStmt that defines the symm pointer
          return LetStmt(symm_src_var, symm_src_ptr_expr, modified_stmt);
        }
      }
    }

    // Default: use parent's visitor
    return StmtExprMutator::VisitStmt_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
};

tvm::transform::Pass DeclareSymmBuffer() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    f = SymmBufferDeclarer::Apply(std::move(f));
    return f;
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0,
                                            "tl.DeclareSymmBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.DeclareSymmBuffer", DeclareSymmBuffer);
});

} // namespace tl
} // namespace tvm
