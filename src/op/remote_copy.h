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

class PutOp : public Operator {
public:
  PutOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<PutOp>(*this);
  }

  PrimExpr get_offset(const BufferLoadNode *load);

private:
  PrimExpr src_addr, dst_addr;
  PrimExpr src_offset, dst_offset;
  PrimExpr copy_size, dst_pe;
  int unroll_factor;
  bool is_symmetric = false;
  Buffer src_buffer, dst_buffer;
  std::string scope; // {warp, block}
};

class GetOp : public Operator {
public:
  GetOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

  std::unique_ptr<Operator> Clone() const final {
    return std::make_unique<GetOp>(*this);
  }

  PrimExpr get_offset(const BufferLoadNode *load);

private:
  PrimExpr src_addr, dst_addr;
  PrimExpr src_offset, dst_offset;
  PrimExpr copy_size, src_pe;
  int unroll_factor;
  bool is_symmetric = false;
  Buffer src_buffer, dst_buffer;
  std::string scope; // {warp, block}
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_BULK_COPY_H_