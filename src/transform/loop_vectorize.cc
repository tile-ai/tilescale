/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file loop_vectorize.cc
 * \brief A tool to automatically vectorize a for loop
 */

#include "loop_vectorize.h"
#include "../op/builtin.h"
#include "../target/utils.h"
#include "arith/int_operator.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_vectorization_utils.h"
#include "tvm/tir/analysis.h"
#include "tvm/tir/var.h"
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tl {

using namespace tir;

struct VectorizePlanResult {
  int vector_size;
  bool dynamic;
  PrimExpr condition;
};

class VectorizeFindGlobalAccess : public arith::IRVisitorWithAnalyzer {
public:
  VectorizeFindGlobalAccess() = default;

  bool HasGlobalAccess(const Stmt &stmt) {
    this->operator()(stmt);
    return has_global_access_;
  }

private:
  bool has_global_access_ = false;

  void VisitStmt_(const BufferStoreNode *node) final {
    if (node->buffer.scope() == "global")
      has_global_access_ = true;
    return arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const BufferLoadNode *node) final {
    if (node->buffer.scope() == "global")
      has_global_access_ = true;
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }
};

class VectorizePlanner : public arith::IRVisitorWithAnalyzer {
public:
  VectorizePlanner() = default;

  int Plan(const For &node) {
    tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
    Optional<Bool> opt_disable_vectorize_256 =
        ctxt->GetConfig(kDisableVectorize256, Optional<Bool>());
    bool disable_vectorize_256 =
        opt_disable_vectorize_256.value_or(Bool(false));
    if (tvm::tl::TargetIsSm100(Target::Current(false)) &&
        !disable_vectorize_256 &&
        VectorizeFindGlobalAccess().HasGlobalAccess(node)) {
      vector_load_bits_max_ = vector_size_ = 256;
    } else {
      vector_load_bits_max_ = vector_size_ = 128;
    }
    this->operator()(node);
    return vector_size_;
  }

private:
  void VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    auto extent_ptr = as_const_int(analyzer_.Simplify(node->extent));
    // Here I disable dynamic shape completely,
    //   In order to do it, the Planner should accept an analyzer with
    //   arithmetic info outside to prove the dividiblity of vector size
    if (!extent_ptr) {
      vector_size_ = 1;
      return;
    }
    vector_size_ = arith::ZeroAwareGCD(vector_size_, *extent_ptr);
    arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const BufferLoadNode *node) final {
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    if (node->buffer->shape.size() == 1) {
      // TODO(lei): This should be improved as
      // constant buffer that tl hack to use as local register.
      auto boundary_check = node->buffer->shape[0].as<IntImmNode>();
      if (boundary_check && boundary_check->value == 1) {
        return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
      }
    }
    UpdateVectorSize(node->indices, node->buffer);
  }

  void VisitStmt_(const BufferStoreNode *node) final {
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    UpdateVectorSize(node->indices, node->buffer);
    return arith::IRVisitorWithAnalyzer::VisitExpr(node->value);
  }

  void VisitStmt_(const IfThenElseNode *node) final {
    CheckConditionVectorized(node->condition);
    return arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const CallNode *node) final {
    if (node->op == builtin::if_then_else()) {
      CheckConditionVectorized(node->args[0]);
    } else if (node->op == builtin::call_extern()) {
      // do not vectorize extern calls
      vector_size_ = 1;
    }
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void CheckConditionVectorized(const PrimExpr &cond) {
    // TODO: perform some checks here
  }

  void VisitExpr_(const CastNode *node) final {
    vector_size_ = arith::ZeroAwareGCD(
        vector_load_bits_max_ / node->dtype.bits(), vector_size_);
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void UpdateVectorSize(const Array<PrimExpr> indices, const Buffer &buffer) {
    if (!inner_for_)
      return;
    // 1. Compute raw element offset
    auto strides = buffer->strides;
    if (buffer->strides.empty()) {
      PrimExpr stride = 1;
      for (int i = indices.size() - 1; i >= 0; --i) {
        strides.push_back(stride);
        stride = stride * buffer->shape[i];
      }
      strides = Array<PrimExpr>{strides.rbegin(), strides.rend()};
    }
    PrimExpr elem_offset = 0;
    for (int i = 0; i < indices.size(); ++i) {
      elem_offset += indices[i] * strides[i];
    }

    // 2. If element offset is independent with loop_var, ignore it
    if (CanProveIndependent(elem_offset, inner_for_->loop_var, &analyzer_)) {
      return;
    }

    // 3. Tight vectorize bound
    vector_size_ = arith::ZeroAwareGCD(vector_size_, vector_load_bits_max_ /
                                                         buffer->dtype.bits());

    // 4. Try to vectorize buffer load
    while (!IndiceCanVectorize(elem_offset, inner_for_->loop_var,
                               inner_for_->extent, vector_size_, &analyzer_)) {
      vector_size_ /= 2;
    }
  }

  int vector_load_bits_max_;

  const ForNode *inner_for_{};
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 128;
};

class VectorizeRewriter : public StmtExprMutator {
public:
  VectorizeRewriter(int vector_size) : vector_size_(vector_size) {}

private:
  Stmt VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    auto ret = StmtExprMutator::VisitStmt_(node);
    if (inner_for_ == node) { // rewrite the innermost loop
      For fnode = ret.as<For>().value();
      auto old_var = fnode->loop_var;
      auto extent_ptr = as_const_int(fnode->extent);
      ICHECK(extent_ptr) << fnode->extent;
      int extent = *extent_ptr;
      ICHECK(extent % vector_size_ == 0)
          << "extent: " << extent << " vector_size_: " << vector_size_;
      ICHECK(is_zero(fnode->min));
      if (extent == vector_size_) {
        fnode.CopyOnWrite()->kind = ForKind::kVectorized;
        return fnode;
      } else {
        Var inner_var = Var("vec");
        Var outer_var = Var(old_var->name_hint);
        Map<Var, PrimExpr> vmap;
        vmap.Set(fnode->loop_var, outer_var * vector_size_ + inner_var);
        Stmt body = Substitute(fnode->body, vmap);
        body = For(inner_var, 0, vector_size_, ForKind::kVectorized, body);
        body = For(outer_var, 0, extent / vector_size_, fnode->kind, body,
                   fnode->thread_binding, fnode->annotations, fnode->span);
        return body;
      }
    } else {
      return ret;
    }
  }

  const ForNode *inner_for_{};
  const int vector_size_;
};

int GetVectorizeSize(const For &loop) { return VectorizePlanner().Plan(loop); }

bool CanProveIndependent(const PrimExpr &expr, Var var,
                         arith::Analyzer *analyzer) {
  // 1. if var doesn't exist, it is independent
  bool used_var = UsesVar(
      expr, [&](const VarNode *v) { return GetRef<Var>(v).same_as(var); });
  if (!used_var) {
    return true;
  }
  // 2. if \forall v_1, v_2, f(v_1) == f(v_2), f is independent with v
  Var var_1("_t", var.dtype());
  auto expr_1 = Substitute(expr, {{var, var_1}});
  if (analyzer->CanProveEqual(expr, expr_1)) {
    return true;
  }
  return false;
}

bool IndiceCanVectorize(const PrimExpr &expr, Var var,
                        const PrimExpr &iter_var_size,
                        int target_vectorized_size, arith::Analyzer *analyzer) {
  ICHECK(target_vectorized_size >= 1);
  if (target_vectorized_size == 1)
    return true;

  // Extent must be divisible
  if (!analyzer->CanProveEqual(FloorMod(iter_var_size, target_vectorized_size),
                               0))
    return false;

  // The base offset must be divisible
  if (!analyzer->CanProveEqual(
          FloorMod(Substitute(expr, {{var, 0}}), target_vectorized_size), 0)) {
    return false;
  }

  // Bind thread range
  Var v0("v0"), v1("v1");
  analyzer->Bind(v0, Range(0, target_vectorized_size));
  analyzer->Bind(v1, Range(0, analyzer->Simplify(FloorDiv(
                                  iter_var_size, target_vectorized_size))));
  PrimExpr expr_transformed = analyzer->Simplify(
      Substitute(expr, {{var, v0 + v1 * target_vectorized_size}}));
  Vectorizer vectorizer(v0, IntImm(v0->dtype, target_vectorized_size));
  PrimExpr expr_vectorized = vectorizer.VisitExpr(expr_transformed);

  // This simplify is necessary for thread region specified
  // optimizations.
  expr_vectorized = analyzer->Simplify(expr_vectorized);
  auto ramp_node = expr_vectorized.as<RampNode>();
  if (!ramp_node) {
    // Broadcast value
    if (expr_vectorized.dtype().lanes() == 1)
      return true;
    else
      return false;
  } else {
    return is_one(ramp_node->stride);
  }
}

For VectorizeLoop(const For &loop, int vectorize_hint) {
  if (vectorize_hint <= 0) {
    VectorizePlanner planner;
    vectorize_hint = planner.Plan(loop);
  }
  if (vectorize_hint == 1)
    return loop;
  auto rewriter = VectorizeRewriter(vectorize_hint);
  return Downcast<For>(rewriter(loop));
}

} // namespace tl
} // namespace tvm
