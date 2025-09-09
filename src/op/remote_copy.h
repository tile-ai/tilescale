/*!
 * \file tl/op/remote_copy.h
 * \brief Warp-level remote copy operators.
 *
 */

#ifndef TVM_TL_OP_BULK_COPY_H_
#define TVM_TL_OP_BULK_COPY_H_

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include "../layout/layout.h"
#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class PushWarpOp : public Operator {
public:
  PushWarpOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<PushWarpOp>(*this);
  }

  PrimExpr get_offset(const BufferLoadNode *load);

private:
  PrimExpr src_addr, dst_addr;
  PrimExpr src_offset, dst_offset;
  PrimExpr copy_size, dst_pe;
  int unroll_factor;
  bool is_symmetric = false;
  Buffer src_buffer, dst_buffer;
};

class PullWarpOp : public Operator {
public:
  PullWarpOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<PullWarpOp>(*this);
  }

  PrimExpr get_offset(const BufferLoadNode *load);

private:
  PrimExpr src_addr, dst_addr;
  PrimExpr src_offset, dst_offset;
  PrimExpr copy_size, src_pe;
  int unroll_factor;
  bool is_symmetric = false;
  Buffer src_buffer, dst_buffer;
};

class PushBlockOp : public Operator {
public:
  PushBlockOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<PushBlockOp>(*this);
  }

  PrimExpr get_offset(const BufferLoadNode *load);

private:
  PrimExpr src_addr, dst_addr;
  PrimExpr src_offset, dst_offset;
  PrimExpr copy_size, dst_pe;
  bool is_symmetric = false;
  Buffer src_buffer, dst_buffer;
};

class PullBlockOp : public Operator {
public:
  PullBlockOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<PullBlockOp>(*this);
  }

  PrimExpr get_offset(const BufferLoadNode *load);

private:
  PrimExpr src_addr, dst_addr;
  PrimExpr src_offset, dst_offset;
  PrimExpr copy_size, src_pe;
  bool is_symmetric = false;
  Buffer src_buffer, dst_buffer;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_BULK_COPY_H_