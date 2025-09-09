/*!
 * \file tl/op/remote_copy.h
 * \brief Remote copy operators.
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

class PutWarpOp : public Operator {
public:
  PutWarpOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<PutWarpOp>(*this);
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

class GetWarpOp : public Operator {
public:
  GetWarpOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<GetWarpOp>(*this);
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

class PutBlockOp : public Operator {
public:
  PutBlockOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<PutBlockOp>(*this);
  }

  PrimExpr get_offset(const BufferLoadNode *load);

private:
  PrimExpr src_addr, dst_addr;
  PrimExpr src_offset, dst_offset;
  PrimExpr copy_size, dst_pe;
  bool is_symmetric = false;
  Buffer src_buffer, dst_buffer;
};

class GetBlockOp : public Operator {
public:
  GetBlockOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<GetBlockOp>(*this);
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