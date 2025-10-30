// TileScale pass

/*!
 * \file declare_symm_buffer.cc
 * \brief Declare the symmetry buffer to prepare for operators that need buffers on peer's symm heap
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

PrimExpr CalculateSymmPtr(PrimExpr ptr, PrimExpr pe) {
  PrimExpr local_rank = Call(DataType::Int(64), tl::get_rank(), {});
  PrimExpr local_base_ptr =
      Call(DataType::Handle(), tl::get_local_base_ptr(), {});
  PrimExpr offset_to_base =
      Sub(Call(DataType::Handle(), tl::get_uintptr_t(), {ptr}),
          local_base_ptr);
  PrimExpr result = Call(DataType::Handle(), tl::get_remote_base_ptr(), {pe}) + offset_to_base;
  return result;
}

/*!
 * \brief Declare the symmetry buffer to prepare for operators that need buffers on peer's symm heap
 */
class SymmBufferDeclarer : public StmtExprMutator {
public:
  static PrimFunc Apply(PrimFunc f) {
    if (!f->body.defined()) {
      return f;
    }

    SymmBufferDeclarer declarer;

    // Extract symm buffer info and replace them
    // The LetStmt insertion will happen inside VisitStmt_(const BlockNode*)
    f.CopyOnWrite()->body = declarer.VisitStmt(f->body);

    return f;
  };

private:
  // Override BlockNode visitor to insert LetStmt inside blocks, not at PrimFunc level
  Stmt VisitStmt_(const BlockNode *op) final {
    // First, recursively visit children to collect let_bindings
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    
    // Insert let bindings inside the block body (not at PrimFunc level)
    // We do this after visiting to ensure all let_bindings are collected
    if (!let_bindings_.empty() && !let_bindings_inserted_) {
      // Insert inside any non-root block to avoid PrimFunc-level insertion
      // The "tilelang_root" or similar computation blocks are ideal
      if (op->name_hint != "root") {
        let_bindings_inserted_ = true;
        Stmt body = block->body;
        // Wrap the block body with all let bindings
        for (const auto& kv : let_bindings_) {
          body = LetStmt(GetRef<Var>(kv.first), kv.second, body);
        }
        BlockNode* n = block.CopyOnWrite();
        n->body = body;
      }
    }
    
    return block;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    // LOG(INFO) << "Found call";
    auto parsed_op = ParseOperator(GetRef<Call>(op), buffer_data_to_buffer_);
    if (parsed_op.defined() && parsed_op.as<CopyNode>()) {
      // LOG(INFO) << "Found copy";
      if (parsed_op.as<CopyNode>()->is_remote_copy) {
        // LOG(INFO) << "Found remote copy";
        if (parsed_op.as<CopyNode>()->dst_pe.defined()) // TODO: add check here
          // && parsed_op.as<CopyNode>()->dst_pe.as<IntImmNode>()->value != -1) 
        {
          LOG(INFO) << "Found remote push";
          Buffer dst = parsed_op.as<CopyNode>()->dst;
          Array<Range> dst_range = parsed_op.as<CopyNode>()->dst_range;

          // 1. Calculate symm dst ptr
          PrimExpr symm_dst_ptr_expr = CalculateSymmPtr(dst->data, parsed_op.as<CopyNode>()->dst_pe);
          LOG(INFO) << "Symm dst ptr expr: " << symm_dst_ptr_expr;

          // 2. Record a let stmt to assign PrimExpr to Var
          String storage_scope = dst->data->type_annotation.as<PointerTypeNode>()->storage_scope;
          Var symm_dst_var = Var(dst->name+"_symm", PointerType(PrimType(dst->dtype), storage_scope));
          PrimExpr casted_ptr = Cast(DataType::Handle(), 
            symm_dst_ptr_expr);
          let_bindings_[symm_dst_var.get()] = casted_ptr;
          
          // 3. Create modified dst buffer with symm var
          dst.CopyOnWrite()->data = symm_dst_var;

          // 4. Rebuild the destination region call with the modified buffer
          // RegionOp args: [BufferLoad(min_indices), access_mask, extent_0, extent_1, ...]
          Array<PrimExpr> dst_region_mins;
          Array<PrimExpr> dst_region_extents;
          for (const Range& r : dst_range) {
            dst_region_mins.push_back(r->min);
            dst_region_extents.push_back(r->extent);
          }
          BufferLoad dst_load(dst, dst_region_mins);
          
          Array<PrimExpr> dst_region_args;
          dst_region_args.push_back(dst_load);
          dst_region_args.push_back(IntImm(DataType::Int(32), op->args[1].as<CallNode>()->args[1].as<IntImmNode>()->value)); // access_mask
          for (const PrimExpr& extent : dst_region_extents) {
            dst_region_args.push_back(extent);
          }
          
          // Create new Call for the destination region
          Call dst_region_call = Call(op->args[1].as<CallNode>()->dtype, 
                                      op->args[1].as<CallNode>()->op, 
                                      dst_region_args, 
                                      op->args[1].as<CallNode>()->span);

          // 5. Rebuild the Copy call with modified args
          Array<PrimExpr> new_copy_args;
          new_copy_args.push_back(op->args[0]); // src region (unchanged)
          new_copy_args.push_back(dst_region_call); // modified dst region
          // Copy remaining args
          for (size_t i = 2; i < op->args.size(); i++) {
            new_copy_args.push_back(op->args[i]);
          }
          
          return Call(op->dtype, op->op, new_copy_args, op->span);
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_map<const VarNode *, PrimExpr> let_bindings_;
  bool let_bindings_inserted_ = false;
};

tvm::transform::Pass DeclareSymmBuffer() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    f = SymmBufferDeclarer::Apply(std::move(f));
    return f;
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0, "tl.DeclareSymmBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.DeclareSymmBuffer", DeclareSymmBuffer);
});

} // namespace tl
} // namespace tvm
