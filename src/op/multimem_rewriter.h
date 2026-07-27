/*!
 * \file tl/op/multimem_rewriter.h
 * \brief Post-process IR to replace vectorized BufferLoad/Store on mcast
 * buffers with multimem call_extern instructions.
 */

#ifndef TVM_TL_OP_MULTIMEM_REWRITER_H_
#define TVM_TL_OP_MULTIMEM_REWRITER_H_

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <sstream>

#include "multimem.h"

namespace tvm {
namespace tl {

using namespace tirx;

static inline std::string GetReduceOpStr(int reduce_op) {
  switch (reduce_op) {
  case 0:
    return "tl::multimem::ReduceOp::ADD";
  case 1:
    return "tl::multimem::ReduceOp::MIN";
  case 2:
    return "tl::multimem::ReduceOp::MAX";
  default:
    LOG(FATAL) << "Invalid reduce_op: " << reduce_op;
    return "";
  }
}

class PlainMulticastAccessFinder : public StmtExprVisitor {
public:
  explicit PlainMulticastAccessFinder(Buffer target)
      : target_(std::move(target)) {}

  bool Find(const Stmt &stmt) {
    VisitStmt(stmt);
    return found_;
  }

private:
  Buffer target_;
  bool found_{false};

  void VisitStmt_(const BufferStoreNode *op) override {
    if (op->buffer.same_as(target_)) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) override {
    if (op->buffer.same_as(target_)) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) override {
    // BufferLoad under address_of denotes an address; it is not a plain load.
    if (op->op.same_as(builtin::address_of())) {
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

/*!
 * \brief Rewrites BufferLoad/BufferStore involving a multicast buffer
 *        into multimem call_extern instructions.
 *
 * After ParallelOp + VectorizeLoop, the IR contains ForKind::kVectorized loops
 * with scalar loop variables (Ramp is not materialized until codegen).
 * This rewriter detects two patterns:
 *
 * 1. ForKind::kVectorized loop containing mcast buffer access:
 *    for (vec: kVectorized, extent=N) { dst[base+vec] = mcast[base+vec] }
 *    → call_extern("LdReduceVN<...>::run", &dst[base], &mcast[base])
 *
 * 2. Scalar BufferStore with Ramp indices (if vectorization produced Ramp):
 *    dst[Ramp(base,1,N)] = mcast[Ramp(base,1,N)]
 *    → call_extern("LdReduceVN<...>::run", &dst[base], &mcast[base])
 */
class MultimemRewriter : public StmtExprMutator {
public:
  MultimemRewriter(Buffer mcast_buf, MultimemMode mode, int reduce_op)
      : mcast_buf_(std::move(mcast_buf)), mode_(mode), reduce_op_(reduce_op) {}

  Stmt Rewrite(Stmt stmt) {
    Stmt result = VisitStmt(std::move(stmt));
    ICHECK(!PlainMulticastAccessFinder(mcast_buf_).Find(result))
        << "multimem lowering left a plain multicast BufferLoad/BufferStore; "
           "this access pattern cannot be represented safely:\n"
        << result;
    return result;
  }

protected:
  /*!
   * \brief Handle ForKind::kVectorized loops.
   * If the loop body is a single BufferStore involving the mcast buffer,
   * replace the entire loop with a single vectorized multimem call.
   */
  Stmt VisitStmt_(const ForNode *op) override {
    if (op->kind == ForKind::kVectorized) {
      auto extent_ptr = op->extent.as<IntImmNode>();
      if (extent_ptr) {
        int lanes = static_cast<int>(extent_ptr->value);
        // Try to match the loop body as a single BufferStore from mcast buffer
        auto result = TryRewriteVectorizedLoop(op, lanes);
        if (result.defined()) {
          return result;
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  /*!
   * \brief Handle scalar BufferStore and Ramp-indexed vector accesses.
   */
  Stmt VisitStmt_(const BufferStoreNode *op) override {
    if (mode_ == MultimemMode::kLdReduce) {
      const BufferLoadNode *load = nullptr;
      Optional<PrimExpr> source_predicate;
      if (MatchLdReduceValue(op->value, &load, &source_predicate) &&
          load->buffer.same_as(mcast_buf_)) {
        int local_lanes = GetLanes(op->indices);
        int mcast_lanes = GetLanes(load->indices);
        ICHECK_EQ(local_lanes, mcast_lanes)
            << "multimem load-reduce requires matching vector lanes";
        if (!source_predicate.defined()) {
          return MakeRampMultimemCalls(op->buffer, op->indices, load->buffer,
                                       load->indices, local_lanes);
        }

        Array<Stmt> scalar_calls;
        for (int i = 0; i < local_lanes; ++i) {
          Array<PrimExpr> local_indices = ExtractLaneIndices(op->indices, i);
          Array<PrimExpr> mcast_indices = ExtractLaneIndices(load->indices, i);
          PrimExpr lane_valid = ExtractLane(source_predicate.value(), i);
          Stmt call = MakeMultimemCall(op->buffer, local_indices, load->buffer,
                                       mcast_indices, 1);
          Stmt zero = BufferStore(op->buffer, make_zero(op->buffer->dtype),
                                  local_indices);
          scalar_calls.push_back(IfThenElse(lane_valid, call, zero));
        }
        return scalar_calls.size() == 1 ? scalar_calls[0]
                                        : SeqStmt(std::move(scalar_calls));
      }
    } else {
      if (op->buffer.same_as(mcast_buf_)) {
        if (auto *load = op->value.as<BufferLoadNode>()) {
          int local_lanes = GetLanes(load->indices);
          int mcast_lanes = GetLanes(op->indices);
          ICHECK_EQ(local_lanes, mcast_lanes)
              << "multimem store/reduce requires matching vector lanes";
          return MakeRampMultimemCalls(load->buffer, load->indices, op->buffer,
                                       op->indices, local_lanes);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  Buffer mcast_buf_;
  MultimemMode mode_;
  int reduce_op_;

  bool IsZeroValue(const PrimExpr &value) const {
    if (is_zero(value)) {
      return true;
    }
    if (auto *float_imm = value.as<FloatImmNode>()) {
      return float_imm->value == 0.0;
    }
    return false;
  }

  bool MatchLdReduceValue(const PrimExpr &value, const BufferLoadNode **load,
                          Optional<PrimExpr> *predicate) const {
    if (auto *direct_load = value.as<BufferLoadNode>()) {
      *load = direct_load;
      *predicate = std::nullopt;
      return true;
    }
    if (auto *call = value.as<CallNode>()) {
      if (call->op.same_as(builtin::if_then_else()) && call->args.size() == 3 &&
          IsZeroValue(call->args[2])) {
        if (auto *guarded_load = call->args[1].as<BufferLoadNode>()) {
          *load = guarded_load;
          *predicate = call->args[0];
          return true;
        }
      }
    }
    return false;
  }

  /*!
   * \brief Try to rewrite a kVectorized for-loop containing a mcast
   * BufferStore. Returns the replacement Stmt, or undefined if the pattern
   * doesn't match.
   */
  Stmt TryRewriteVectorizedLoop(const ForNode *op, int lanes) {
    // The body should be a single BufferStore, possibly wrapped in one or more
    // predicates without else branches.
    const BufferStoreNode *store = nullptr;
    Stmt body = op->body;
    Array<PrimExpr> predicates;
    while (auto *ite = body.as<IfThenElseNode>()) {
      if (ite->else_case.defined()) {
        return Stmt();
      }
      predicates.push_back(ite->condition);
      body = ite->then_case;
    }
    store = body.as<BufferStoreNode>();

    if (!store)
      return Stmt();

    const BufferLoadNode *load = nullptr;
    Optional<PrimExpr> source_predicate;
    if (mode_ == MultimemMode::kLdReduce) {
      if (!MatchLdReduceValue(store->value, &load, &source_predicate)) {
        return Stmt();
      }
    } else {
      load = store->value.as<BufferLoadNode>();
      if (!load)
        return Stmt();
    }

    // Check if this involves the mcast buffer
    bool matches = false;
    const Buffer *local_buf_ptr = nullptr;
    const Array<PrimExpr> *local_indices_ptr = nullptr;
    const Buffer *mc_buf_ptr = nullptr;
    const Array<PrimExpr> *mc_indices_ptr = nullptr;

    if (mode_ == MultimemMode::kLdReduce) {
      if (load->buffer.same_as(mcast_buf_)) {
        matches = true;
        local_buf_ptr = &store->buffer;
        local_indices_ptr = &store->indices;
        mc_buf_ptr = &load->buffer;
        mc_indices_ptr = &load->indices;
      }
    } else {
      if (store->buffer.same_as(mcast_buf_)) {
        matches = true;
        local_buf_ptr = &load->buffer;
        local_indices_ptr = &load->indices;
        mc_buf_ptr = &store->buffer;
        mc_indices_ptr = &store->indices;
      }
    }

    if (!matches)
      return Stmt();

    // Split widths larger than 128 bits into supported V4/V2/V1 calls.  This
    // handles targets where the generic vectorizer selects V8 f32.
    Var vec_var = op->loop_var;
    Array<Stmt> vector_calls;
    int lane_offset = 0;
    while (lane_offset < lanes) {
      int chunk_lanes = NextSupportedWidth(lanes - lane_offset);
      PrimExpr lane = op->min + IntImm(vec_var.dtype(), lane_offset);
      vector_calls.push_back(MakeMultimemCall(
          *local_buf_ptr, SubstituteIndices(*local_indices_ptr, vec_var, lane),
          *mc_buf_ptr, SubstituteIndices(*mc_indices_ptr, vec_var, lane),
          chunk_lanes));
      lane_offset += chunk_lanes;
    }
    Stmt vector_path = vector_calls.size() == 1
                           ? vector_calls[0]
                           : SeqStmt(std::move(vector_calls));

    if (predicates.empty() && !source_predicate.defined()) {
      return vector_path;
    }

    // A wide call is legal only if every original predicate holds for every
    // lane.  Otherwise retain the predicates and issue scalar multimem calls.
    PrimExpr all_lanes_valid = make_const(DataType::Bool(), true);
    Array<Stmt> scalar_calls;
    for (int i = 0; i < lanes; ++i) {
      PrimExpr lane = op->min + IntImm(vec_var.dtype(), i);
      Map<Var, PrimExpr> vmap;
      vmap.Set(vec_var, lane);
      PrimExpr statement_valid = make_const(DataType::Bool(), true);
      for (const auto &predicate : predicates) {
        statement_valid =
            And(statement_valid, ExtractLane(Substitute(predicate, vmap), i));
      }
      PrimExpr source_valid = make_const(DataType::Bool(), true);
      if (source_predicate.defined()) {
        source_valid =
            ExtractLane(Substitute(source_predicate.value(), vmap), i);
      }
      all_lanes_valid =
          And(all_lanes_valid, And(statement_valid, source_valid));

      Array<PrimExpr> local_indices =
          SubstituteIndices(*local_indices_ptr, vec_var, lane);
      Stmt scalar_call = MakeMultimemCall(
          *local_buf_ptr, local_indices, *mc_buf_ptr,
          SubstituteIndices(*mc_indices_ptr, vec_var, lane), 1);
      Stmt lane_action = scalar_call;
      if (source_predicate.defined()) {
        Stmt zero = BufferStore(
            *local_buf_ptr, make_zero((*local_buf_ptr)->dtype), local_indices);
        lane_action = IfThenElse(source_valid, scalar_call, zero);
      }
      if (!predicates.empty()) {
        lane_action = IfThenElse(statement_valid, lane_action);
      }
      scalar_calls.push_back(lane_action);
    }
    Stmt scalar_path = scalar_calls.size() == 1
                           ? scalar_calls[0]
                           : SeqStmt(std::move(scalar_calls));
    return IfThenElse(all_lanes_valid, vector_path, scalar_path);
  }

  static int NextSupportedWidth(int remaining) {
    if (remaining >= 4)
      return 4;
    if (remaining >= 2)
      return 2;
    return 1;
  }

  Array<PrimExpr> SubstituteIndices(const Array<PrimExpr> &indices,
                                    const Var &var,
                                    const PrimExpr &value) const {
    Map<Var, PrimExpr> vmap;
    vmap.Set(var, value);
    Array<PrimExpr> result;
    for (const auto &index : indices) {
      result.push_back(Substitute(index, vmap));
    }
    return result;
  }

  /*!
   * \brief Get the vector lanes from Ramp indices.
   */
  int GetLanes(const Array<PrimExpr> &indices) const {
    int lanes = 1;
    for (const auto &index : indices) {
      int index_lanes = index.dtype().lanes();
      if (index_lanes > 1) {
        ICHECK(lanes == 1 || lanes == index_lanes)
            << "multimem indices have inconsistent vector lanes";
        lanes = index_lanes;
      }
    }
    return lanes;
  }

  Array<PrimExpr> ExtractLaneIndices(const Array<PrimExpr> &indices,
                                     int lane_offset) const {
    Array<PrimExpr> result;
    for (const auto &index : indices) {
      if (auto *ramp = index.as<RampNode>()) {
        result.push_back(ramp->base + ramp->stride * lane_offset);
      } else if (auto *broadcast = index.as<BroadcastNode>()) {
        result.push_back(broadcast->value);
      } else if (index.dtype().lanes() > 1) {
        result.push_back(Shuffle::ExtractElement(index, lane_offset));
      } else {
        result.push_back(index);
      }
    }
    return result;
  }

  PrimExpr ExtractLane(const PrimExpr &value, int lane) const {
    if (auto *broadcast = value.as<BroadcastNode>()) {
      return broadcast->value;
    }
    if (value.dtype().lanes() > 1) {
      return Shuffle::ExtractElement(value, lane);
    }
    return value;
  }

  Stmt MakeRampMultimemCalls(const Buffer &local_buf,
                             const Array<PrimExpr> &local_indices,
                             const Buffer &mc_buf,
                             const Array<PrimExpr> &mc_indices,
                             int lanes) const {
    Array<Stmt> calls;
    int lane_offset = 0;
    while (lane_offset < lanes) {
      int chunk_lanes = NextSupportedWidth(lanes - lane_offset);
      calls.push_back(MakeMultimemCall(
          local_buf, ExtractLaneIndices(local_indices, lane_offset), mc_buf,
          ExtractLaneIndices(mc_indices, lane_offset), chunk_lanes));
      lane_offset += chunk_lanes;
    }
    return calls.size() == 1 ? calls[0] : SeqStmt(std::move(calls));
  }

  /*!
   * \brief Create the call_extern for a multimem instruction.
   */
  Stmt MakeMultimemCall(const Buffer &local_buf,
                        const Array<PrimExpr> &local_indices,
                        const Buffer &mc_buf, const Array<PrimExpr> &mc_indices,
                        int lanes) const {
    std::string func_name = MakeFuncName(lanes, local_buf->dtype);

    Array<PrimExpr> args;
    args.push_back(StringImm(func_name));

    if (mode_ == MultimemMode::kLdReduce) {
      args.push_back(MakeAddressOf(local_buf, local_indices));
      args.push_back(MakeAddressOf(mc_buf, mc_indices));
    } else {
      args.push_back(MakeAddressOf(mc_buf, mc_indices));
      args.push_back(MakeAddressOf(local_buf, local_indices));
    }

    auto call = Call(DataType::Handle(), builtin::call_extern(), args);
    return Evaluate(call);
  }

  /*!
   * \brief Construct the template function name.
   */
  std::string MakeFuncName(int lanes, DataType dtype) const {
    std::string dtype_tag = DTypeToTag(dtype);
    std::string reduce_op_str;
    if (mode_ != MultimemMode::kSt) {
      reduce_op_str = GetReduceOpStr(reduce_op_);
    }

    std::stringstream ss;
    switch (mode_) {
    case MultimemMode::kLdReduce:
      ss << "tl::multimem::LdReduceV" << lanes;
      break;
    case MultimemMode::kSt:
      ss << "tl::multimem::StV" << lanes;
      break;
    case MultimemMode::kRed:
      ss << "tl::multimem::RedV" << lanes;
      break;
    }
    ss << "<";
    if (mode_ != MultimemMode::kSt) {
      ss << reduce_op_str << ", ";
    }
    ss << dtype_tag;
    ss << ">::run";
    return ss.str();
  }

  std::string DTypeToTag(DataType dtype) const {
    if (dtype.lanes() == 1 && dtype.is_float() && dtype.bits() == 32)
      return "float";
    if (dtype.lanes() == 1 && dtype.is_float16())
      return "half_t";
    if (dtype.lanes() == 1 && dtype.is_bfloat16())
      return "bfloat16_t";
    LOG(FATAL) << "Unsupported dtype for multimem: " << dtype;
    return "";
  }

  /*!
   * \brief Create address_of expression. Handles Ramp by extracting base.
   */
  PrimExpr MakeAddressOf(const Buffer &buffer,
                         const Array<PrimExpr> &indices) const {
    Array<PrimExpr> scalar_indices;
    for (const auto &idx : indices) {
      if (auto *ramp = idx.as<RampNode>()) {
        scalar_indices.push_back(ramp->base);
      } else {
        scalar_indices.push_back(idx);
      }
    }
    return Call(DataType::Handle(), builtin::address_of(),
                {BufferLoad(buffer, scalar_indices)});
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_MULTIMEM_REWRITER_H_
