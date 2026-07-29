/*!
 * \file tl/op/multimem.cc
 * \brief Unified multimem operator implementation.
 *
 * Direct modes reuse CopyNode's ParallelOp + InferLayout + VectorizeLoop
 * pipeline before replacing multicast accesses with multimem instructions.
 * Bulk modes validate and lower one contiguous shared-to-multicast region.
 */

#include "multimem.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <limits>
#include <sstream>

#include "../transform/common/loop_fusion_utils.h"
#include "../transform/common/loop_parallel_transform_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "multimem_rewriter.h"
#include "operator.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace multimem_detail {

std::string DTypeToTag(DataType dtype) {
  if (dtype.lanes() == 1 && dtype.is_float() && dtype.bits() == 32)
    return "float";
  if (dtype.lanes() == 1 && dtype.is_float16())
    return "half_t";
  if (dtype.lanes() == 1 && dtype.is_bfloat16())
    return "bfloat16_t";
  LOG(FATAL) << "Unsupported dtype for multimem: " << dtype;
  return "";
}

std::string ReduceOpToTag(int reduce_op) {
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

std::string FuncName(MultimemMode mode, int reduce_op, int lanes,
                     DataType dtype) {
  std::stringstream ss;
  switch (mode) {
  case MultimemMode::kLdReduce:
    ss << "tl::multimem::LdReduceV" << lanes << "<" << ReduceOpToTag(reduce_op)
       << ", " << DTypeToTag(dtype) << ">::run";
    break;
  case MultimemMode::kSt:
    ss << "tl::multimem::StV" << lanes << "<" << DTypeToTag(dtype) << ">::run";
    break;
  case MultimemMode::kRed:
    ss << "tl::multimem::RedV" << lanes << "<" << ReduceOpToTag(reduce_op)
       << ", " << DTypeToTag(dtype) << ">::run";
    break;
  default:
    LOG(FATAL) << "Unsupported multimem mode for vector instruction: "
               << static_cast<int>(mode);
  }
  return ss.str();
}

PrimExpr MakeAddress(const Buffer &buffer, const Array<PrimExpr> &indices) {
  return Call(DataType::Handle(), builtin::address_of(),
              {BufferLoad(buffer, indices)});
}

} // namespace multimem_detail

namespace {

PrimExpr ProductExtent(const Array<Range> &ranges, size_t begin, size_t end) {
  PrimExpr result = 1;
  for (size_t i = begin; i < end; ++i) {
    result = result * ranges[i]->extent;
  }
  return result;
}

PrimExpr FlattenIndices(const Array<PrimExpr> &indices,
                        const Array<PrimExpr> &shape,
                        arith::Analyzer *analyzer) {
  ICHECK_EQ(indices.size(), shape.size());
  PrimExpr flat = 0;
  PrimExpr stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    flat = flat + indices[i] * stride;
    stride = stride * shape[i];
  }
  return analyzer->Simplify(flat);
}

Array<PrimExpr> UnflattenIndex(PrimExpr flat, const Array<PrimExpr> &shape,
                               arith::Analyzer *analyzer) {
  Array<PrimExpr> indices;
  PrimExpr remaining = flat;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    indices.insert(indices.begin(),
                   analyzer->Simplify(floormod(remaining, shape[i])));
    remaining = analyzer->Simplify(floordiv(remaining, shape[i]));
  }
  return indices;
}

enum class RegionBoundsPolicy {
  kRequireStatic,
  kAllowWholeTilePredicate,
};

Optional<PrimExpr> AnalyzeRegionBounds(const Buffer &buffer,
                                       const Array<Range> &ranges,
                                       RegionBoundsPolicy policy,
                                       arith::Analyzer *analyzer,
                                       const char *description) {
  ICHECK_EQ(ranges.size(), buffer->shape.size())
      << description << " region rank must match buffer rank";

  Array<PrimExpr> dynamic_conditions;
  for (size_t i = 0; i < ranges.size(); ++i) {
    PrimExpr lower = analyzer->Simplify(ranges[i]->min);
    PrimExpr upper = analyzer->Simplify(ranges[i]->min + ranges[i]->extent);
    PrimExpr conditions[] = {lower >= 0, upper <= buffer->shape[i]};
    PrimExpr region_extent = analyzer->Simplify(ranges[i]->extent);
    bool is_whole_tile_partition =
        analyzer->CanProve(region_extent > 0,
                           arith::ProofStrength::kSymbolicBound) &&
        analyzer->CanProveEqual(FloorMod(lower, region_extent), 0) &&
        analyzer->CanProveEqual(FloorMod(buffer->shape[i], region_extent), 0);
    for (size_t j = 0; j < 2; ++j) {
      if (analyzer->CanProve(conditions[j],
                             arith::ProofStrength::kSymbolicBound)) {
        continue;
      }
      ICHECK(!analyzer->CanProve(Not(conditions[j]),
                                 arith::ProofStrength::kSymbolicBound))
          << description << " region is statically out of bounds; got ["
          << lower << ", " << upper << ") for extent " << buffer->shape[i]
          << " at dimension " << i;

      ICHECK(policy == RegionBoundsPolicy::kAllowWholeTilePredicate &&
             is_whole_tile_partition)
          << description
          << " region must be provably in bounds or use a tile-aligned "
             "all-or-none dynamic partition; got ["
          << lower << ", " << upper << ") for extent " << buffer->shape[i]
          << " at dimension " << i;
      dynamic_conditions.push_back(conditions[j]);
    }
  }

  if (dynamic_conditions.empty()) {
    return std::nullopt;
  }
  PrimExpr predicate = dynamic_conditions[0];
  for (size_t i = 1; i < dynamic_conditions.size(); ++i) {
    predicate = And(predicate, dynamic_conditions[i]);
  }
  return analyzer->Simplify(predicate);
}

void RequireRegionInBounds(const Buffer &buffer, const Array<Range> &ranges,
                           arith::Analyzer *analyzer, const char *description) {
  ICHECK(!AnalyzeRegionBounds(buffer, ranges,
                              RegionBoundsPolicy::kRequireStatic, analyzer,
                              description)
              .defined());
}

void RequireBufferBaseAlignment(const Buffer &buffer, int alignment,
                                const char *description) {
  ICHECK_GE(buffer->data_alignment, alignment)
      << description << " buffer base alignment must be at least " << alignment
      << " bytes";
  ICHECK_EQ(buffer->data_alignment % alignment, 0)
      << description << " buffer base alignment must be a multiple of "
      << alignment << " bytes";
}

Optional<PrimExpr> ValidatePacked16BitRegions(const Buffer &local_buf,
                                              const Array<Range> &local_range,
                                              const Buffer &mcast_buf,
                                              const Array<Range> &mcast_range,
                                              arith::Analyzer *analyzer) {
  ICHECK_EQ(local_range.size(), mcast_range.size())
      << "multimem packed x2 lowering expects matching region rank";
  ICHECK(!local_range.empty())
      << "multimem packed x2 lowering expects a non-scalar region";

  for (size_t i = 0; i < local_range.size(); ++i) {
    ICHECK(
        analyzer->CanProveEqual(local_range[i]->extent, mcast_range[i]->extent))
        << "multimem packed x2 lowering expects matching region extents";
    ICHECK(analyzer->CanProve(local_range[i]->extent > 0,
                              arith::ProofStrength::kSymbolicBound))
        << "multimem packed x2 lowering requires provably positive region "
           "extents, got "
        << local_range[i]->extent << " at dimension " << i;
  }

  RequireRegionInBounds(local_buf, local_range, analyzer,
                        "multimem packed local");
  Optional<PrimExpr> mcast_in_bounds = AnalyzeRegionBounds(
      mcast_buf, mcast_range, RegionBoundsPolicy::kAllowWholeTilePredicate,
      analyzer, "multimem packed multicast");

  const size_t last = local_range.size() - 1;
  PrimExpr last_extent = analyzer->Simplify(local_range[last]->extent);

  for (size_t i = 0; i < local_range.size(); ++i) {
    ICHECK(analyzer->CanProveEqual(local_range[i]->min, 0))
        << "multimem packed x2 lowering requires local regions to start at "
           "zero because non-zero fragment slices need a different thread "
           "ownership mapping; got min "
        << local_range[i]->min << " at dimension " << i;
    ICHECK(analyzer->CanProveEqual(local_range[i]->extent, local_buf->shape[i]))
        << "multimem packed x2 lowering requires the local region to cover "
           "the entire fragment buffer; got extent "
        << local_range[i]->extent << " for buffer extent "
        << local_buf->shape[i] << " at dimension " << i;
  }

  ICHECK(analyzer->CanProveEqual(FloorMod(last_extent, 2), 0))
      << "multimem packed x2 lowering requires the last dimension extent to "
         "be provably divisible by 2, got "
      << last_extent;

  return mcast_in_bounds;
}

void ValidatePacked16BitPhysicalBuffers(const Buffer &local_buf,
                                        const Buffer &mcast_buf,
                                        const Array<Range> &mcast_range,
                                        arith::Analyzer *analyzer) {
  ICHECK_EQ(mcast_buf->shape.size(), mcast_range.size())
      << "multimem packed x2 lowering expects the remapped multicast buffer "
         "rank to match its region";
  RequireBufferBaseAlignment(local_buf, 4, "multimem packed local");
  RequireBufferBaseAlignment(mcast_buf, 4, "multimem packed multicast");

  Array<PrimExpr> local_start_indices;
  for (size_t i = 0; i < local_buf->shape.size(); ++i) {
    local_start_indices.push_back(0);
  }
  Array<PrimExpr> local_start_offsets =
      local_buf->ElemOffset(local_start_indices);
  ICHECK_EQ(local_start_offsets.size(), 1)
      << "multimem packed x2 lowering requires a flat local address";
  PrimExpr local_start_offset = analyzer->Simplify(local_start_offsets[0]);
  ICHECK(analyzer->CanProveEqual(FloorMod(local_start_offset, 2), 0))
      << "multimem packed x2 lowering requires a 4-byte-aligned local start "
         "address, got element offset "
      << local_start_offset;

  Array<PrimExpr> start_indices;
  for (const auto &range : mcast_range) {
    start_indices.push_back(range->min);
  }
  Array<PrimExpr> start_offsets = mcast_buf->ElemOffset(start_indices);
  ICHECK_EQ(start_offsets.size(), 1)
      << "multimem packed x2 lowering requires a flat multicast address";
  PrimExpr start_offset = analyzer->Simplify(start_offsets[0]);
  ICHECK(analyzer->CanProveEqual(FloorMod(start_offset, 2), 0))
      << "multimem packed x2 lowering requires a 4-byte-aligned multicast "
         "start address, got element offset "
      << start_offset;

  const size_t last = mcast_range.size() - 1;
  for (size_t i = 0; i < mcast_range.size(); ++i) {
    Array<PrimExpr> next_indices = start_indices;
    next_indices.Set(i, next_indices[i] + 1);
    Array<PrimExpr> next_offsets = mcast_buf->ElemOffset(next_indices);
    ICHECK_EQ(next_offsets.size(), 1)
        << "multimem packed x2 lowering requires a flat multicast address";
    PrimExpr physical_stride =
        analyzer->Simplify(next_offsets[0] - start_offset);
    if (i == last) {
      ICHECK(analyzer->CanProveEqual(physical_stride, 1))
          << "multimem packed x2 lowering requires contiguous pairs in the "
             "last dimension, got physical stride "
          << physical_stride;
    } else if (!analyzer->CanProveEqual(mcast_range[i]->extent, 1)) {
      ICHECK(analyzer->CanProveEqual(FloorMod(physical_stride, 2), 0))
          << "multimem packed x2 lowering requires an even physical stride "
             "for each varying leading dimension, got stride "
          << physical_stride << " at dimension " << i;
    }
  }
}

Fragment MakePacked16BitLayout(const Buffer &local_buf,
                               const Array<Range> &local_range,
                               const Range &thread_bounds,
                               arith::Analyzer *analyzer) {
  PrimExpr numel = ProductExtent(local_range, 0, local_range.size());
  PrimExpr thread_extent = thread_bounds->extent;
  PrimExpr pair_width = IntImm(DataType::Int(32), 2);
  // Each thread owns whole pairs and every element has exactly one owner, so
  // the local footprint is pair_width * ceil(numel / (thread_extent *
  // pair_width)) and the layout is not replicated at all in the common case.
  // Replication only applies when there are fewer pairs than threads, where the
  // surplus threads have to duplicate an existing pair to keep the fragment
  // defined over the whole thread range.
  PrimExpr pair_count = floordiv(numel, pair_width);
  PrimExpr replicate_extent = analyzer->Simplify(
      max(IntImm(DataType::Int(32), 1),
          floordiv(thread_extent + pair_count - 1, pair_count)));

  Array<PrimExpr> logical_indices;
  for (size_t i = 0; i < local_buf->shape.size(); ++i) {
    logical_indices.push_back(InputPlaceholder(i));
  }
  PrimExpr logical =
      FlattenIndices(logical_indices, local_buf->shape, analyzer);
  PrimExpr pair_id = floordiv(logical, pair_width);
  PrimExpr local_offset =
      analyzer->Simplify(FloorMod(logical, pair_width) +
                         pair_width * floordiv(pair_id, thread_extent));
  PrimExpr thread = analyzer->Simplify(FloorMod(pair_id, thread_extent));

  return Fragment(local_buf->shape, {local_offset}, thread, replicate_extent,
                  std::nullopt)
      ->BindThreadRange(thread_bounds);
}

void RequirePacked16BitLayout(const Layout &layout, const Fragment &expected,
                              arith::Analyzer *analyzer) {
  auto actual_opt = layout.as<Fragment>();
  ICHECK(actual_opt.has_value())
      << "multimem packed x2 lowering requires a fragment layout for its "
         "local buffer";
  Fragment actual = actual_opt.value();

  auto require_equal = [&](const PrimExpr &actual_expr,
                           const PrimExpr &expected_expr,
                           const char *description) {
    ICHECK(analyzer->CanProveEqual(actual_expr, expected_expr))
        << "multimem packed x2 lowering requires the local fragment layout to "
           "preserve canonical pair ownership; mismatched "
        << description << ": got " << actual_expr << ", expected "
        << expected_expr << "\nactual layout: " << actual->DebugOutput()
        << "\nexpected layout: " << expected->DebugOutput();
  };
  auto require_array_equal = [&](const Array<PrimExpr> &actual_exprs,
                                 const Array<PrimExpr> &expected_exprs,
                                 const char *description) {
    ICHECK_EQ(actual_exprs.size(), expected_exprs.size())
        << "multimem packed x2 lowering requires the local fragment layout to "
           "preserve canonical pair ownership; mismatched "
        << description << " rank\nactual layout: " << actual->DebugOutput()
        << "\nexpected layout: " << expected->DebugOutput();
    for (size_t i = 0; i < actual_exprs.size(); ++i) {
      require_equal(actual_exprs[i], expected_exprs[i], description);
    }
  };

  require_array_equal(actual->InputShape(), expected->InputShape(),
                      "logical shape");
  require_array_equal(actual->OutputShape(), expected->OutputShape(),
                      "physical shape");
  require_equal(actual->ReplicateExtent(), expected->ReplicateExtent(),
                "replicate extent");
  require_equal(actual->ThreadExtent(), expected->ThreadExtent(),
                "thread extent");
  // ThreadRange is optional metadata. The mapping and ThreadExtent checks below
  // establish ownership even when an explicitly annotated Fragment is unbound.
  if (actual->ThreadRange().defined()) {
    ICHECK(expected->ThreadRange().defined());
    require_equal(actual->ThreadRange()->min, expected->ThreadRange()->min,
                  "thread range minimum");
    require_equal(actual->ThreadRange()->extent,
                  expected->ThreadRange()->extent, "thread range extent");
  }

  Array<PrimExpr> logical_vars;
  for (size_t i = 0; i < expected->InputDim(); ++i) {
    Var var("multimem_layout_i" + std::to_string(i), DataType::Int(32));
    analyzer->Bind(var, Range(0, expected->InputShape()[i]));
    logical_vars.push_back(var);
  }
  require_array_equal(actual->Forward(logical_vars),
                      expected->Forward(logical_vars), "physical slot mapping");
  Var replicate("multimem_layout_rep", DataType::Int(32));
  require_equal(actual->ForwardThread(logical_vars, replicate),
                expected->ForwardThread(logical_vars, replicate),
                "thread ownership mapping");
}

} // namespace

// === MultimemOp Constructor ===
// args[0]: src region (tl.region call), args[1]: dst region, args[2]: mode,
// args[3]: reduce_op
MultimemOp::MultimemOp(Array<PrimExpr> args,
                       Map<String, ObjectRef> annotations) {
  ObjectPtr<MultimemOpNode> node = tvm::ffi::make_object<MultimemOpNode>();

  // Parse buffer regions using same utility as CopyNode
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto region = NormalizeToBufferRegion(args[i]);
    rgs[i] = region->region;
    bf[i] = region->buffer;
  }
  node->src = bf[0];
  node->dst = bf[1];
  node->src_range = rgs[0];
  node->dst_range = rgs[1];

  ICHECK_EQ(node->src_range.size(), node->dst_range.size())
      << "multimem expects source and destination regions with matching rank";
  ICHECK_EQ(node->src_range.size(), node->src->shape.size())
      << "multimem source region rank must match source buffer rank";
  ICHECK_EQ(node->dst_range.size(), node->dst->shape.size())
      << "multimem destination region rank must match destination buffer rank";

  node->mode = static_cast<MultimemMode>(args[2].as<IntImm>().value()->value);
  node->reduce_op = args[3].as<IntImm>().value()->value;

  ICHECK_EQ(node->src->dtype, node->dst->dtype)
      << "multimem expects matching source and destination dtypes, got "
      << node->src->dtype << " and " << node->dst->dtype;
  if (node->mode != MultimemMode::kTmaStore &&
      node->mode != MultimemMode::kTmaRedStore) {
    bool supported_dtype =
        node->src->dtype.lanes() == 1 &&
        ((node->src->dtype.is_float() && node->src->dtype.bits() == 32) ||
         node->src->dtype.is_float16() || node->src->dtype.is_bfloat16());
    ICHECK(supported_dtype)
        << "direct multimem operations require scalar float32, float16, or "
           "bfloat16 elements, got "
        << node->src->dtype;
  }
  if (node->mode == MultimemMode::kLdReduce ||
      node->mode == MultimemMode::kRed) {
    ICHECK_EQ(node->reduce_op, 0)
        << "direct multimem load-reduce/reduce currently supports ADD only; "
           "MIN/MAX are not valid for the implemented PTX type combinations";
  }

  // Validate buffer scopes based on mode:
  //   ld_reduce: src=global(mcast), dst=local.fragment
  //   st:        src=local.fragment, dst=global(mcast)
  //   red:       src=local.fragment, dst=global(mcast)
  String src_scope = node->src.scope();
  String dst_scope = node->dst.scope();
  switch (node->mode) {
  case MultimemMode::kLdReduce:
    ICHECK(src_scope == "global")
        << "multimem_ld_reduce: src must be global (multicast) buffer, got '"
        << src_scope << "' for buffer '" << node->src->name << "'";
    ICHECK(dst_scope == "local.fragment")
        << "multimem_ld_reduce: dst must be local.fragment buffer, got '"
        << dst_scope << "' for buffer '" << node->dst->name << "'";
    break;
  case MultimemMode::kSt:
    ICHECK(src_scope == "local.fragment")
        << "multimem_st: src must be local.fragment buffer, got '" << src_scope
        << "' for buffer '" << node->src->name << "'";
    ICHECK(dst_scope == "global")
        << "multimem_st: dst must be global (multicast) buffer, got '"
        << dst_scope << "' for buffer '" << node->dst->name << "'";
    break;
  case MultimemMode::kRed:
    ICHECK(src_scope == "local.fragment")
        << "multimem_red: src must be local.fragment buffer, got '" << src_scope
        << "' for buffer '" << node->src->name << "'";
    ICHECK(dst_scope == "global")
        << "multimem_red: dst must be global (multicast) buffer, got '"
        << dst_scope << "' for buffer '" << node->dst->name << "'";
    break;
  case MultimemMode::kTmaStore:
  case MultimemMode::kTmaRedStore:
    ICHECK(src_scope == "shared" || src_scope == "shared.dyn")
        << "multimem_tma_store: src must be shared memory, got '" << src_scope
        << "' for buffer '" << node->src->name << "'";
    ICHECK(dst_scope == "global")
        << "multimem_tma_store: dst must be global (multicast) buffer, got '"
        << dst_scope << "' for buffer '" << node->dst->name << "'";
    break;
  }

  data_ = std::move(node);
}

// === MakeIterVars ===
// Creates loop iteration variables from ranges (skipping dims with extent==1)
Array<IterVar> MultimemOpNode::MakeIterVars() const {
  // Use the range with the higher scope level as basis (same logic as CopyNode)
  auto scope_level = [](const Buffer &b) -> int {
    String s = b.scope();
    if (s == "local.fragment" || s == "local")
      return 2;
    if (s == "shared" || s == "shared.dyn" || s == "shared.tmem")
      return 1;
    return 0;
  };

  int src_level = scope_level(src);
  int dst_level = scope_level(dst);
  bool base_is_src = (src_level >= dst_level);
  const Array<Range> &base_ranges = base_is_src ? src_range : dst_range;

  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < base_ranges.size(); i++) {
    if (is_one(base_ranges[i]->extent))
      continue;
    Var var = Var(std::string{char('i' + idx)}, base_ranges[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, base_ranges[i]->extent), var, IterVarType::kDataPar});
  }
  return loop_vars;
}

// === MakeIndices ===
Array<PrimExpr> MultimemOpNode::MakeIndices(const Array<IterVar> &ivs,
                                            int src_dst) const {
  Array<PrimExpr> indices;
  const Array<Range> &ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      indices.push_back(ranges[i]->min);
    else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }
  return indices;
}

// === MakePredicate ===
PrimExpr MultimemOpNode::MakePredicate(arith::Analyzer *analyzer,
                                       const Array<PrimExpr> &indices,
                                       const Array<PrimExpr> &extents) const {
  ICHECK_EQ(indices.size(), extents.size());
  Array<PrimExpr> cond_list;
  for (size_t i = 0; i < indices.size(); i++) {
    PrimExpr cond = indices[i] < extents[i];
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = indices[i] >= 0;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
  }
  if (cond_list.empty())
    return {};
  PrimExpr result = cond_list[0];
  for (size_t i = 1; i < cond_list.size(); i++)
    result = And(result, cond_list[i]);
  return result;
}

// === MakeSIMTLoop ===
// Creates the element-wise parallel loop: for (i,j): dst[i,j] = src[i,j]
// Let ParallelOp choose the widest legal vector width for the region.  The
// multimem templates support both V4 and V2 f32 forms, and packed 16-bit modes
// are lowered by LowerPacked16Bit instead of this generic path.
For MultimemOpNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  Array<IterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.empty();

  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);

  Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);

  PrimExpr src_predicate = MakePredicate(analyzer, src_indices, src->shape);
  PrimExpr dst_predicate = MakePredicate(analyzer, dst_indices, dst->shape);

  PrimExpr value = BufferLoad(src, src_indices);
  if (src->dtype != dst->dtype)
    value = Cast(dst->dtype, value);
  if (src_predicate.defined())
    value = if_then_else(src_predicate, value, make_zero(dst->dtype));

  Stmt body = BufferStore(dst, value, dst_indices);
  if (dst_predicate.defined())
    body = IfThenElse(dst_predicate, body);

  if (is_scalar) {
    return For(Var("i"), 0, 1, ForKind::kSerial, body);
  }

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body);
  }
  return Downcast<For>(body);
}

// === InferLayout ===
// Delegates to ParallelOp for layout inference (same as
// CopyNode::LowerNormalCopy)
LayoutMap MultimemOpNode::InferLayout(const LayoutInferArgs &T,
                                      InferLevel level) const {
  if (mode == MultimemMode::kTmaStore || mode == MultimemMode::kTmaRedStore) {
    return {};
  }
  if (IsPacked16BitMultimem()) {
    Buffer local_buf = (mode == MultimemMode::kLdReduce) ? dst : src;
    const Array<Range> &local_range =
        (mode == MultimemMode::kLdReduce) ? dst_range : src_range;
    ICHECK(T.analyzer != nullptr);
    Fragment expected = MakePacked16BitLayout(local_buf, local_range,
                                              T.thread_bounds, T.analyzer);
    if (T.layout_map.count(local_buf)) {
      RequirePacked16BitLayout(T.layout_map[local_buf], expected, T.analyzer);
      return {};
    }
    if (mode == MultimemMode::kLdReduce) {
      // Let downstream fragment consumers choose a layout, then validate the
      // resolved layout again in LowerPacked16Bit.
      return {};
    }

    LayoutMap result;
    result.Set(local_buf, expected);
    return result;
  }
  arith::Analyzer analyzer;
  auto par_op = ParallelOp(MakeTransformedSIMTLoop(&analyzer));
  return par_op->InferLayout(T, level);
}

// === Lower ===
// The main lowering path: MakeSIMTLoop -> ParallelOp pipeline ->
// MultimemRewriter
Stmt MultimemOpNode::Lower(const LowerArgs &T,
                           arith::Analyzer *analyzer) const {
  ICHECK_EQ(src_range.size(), dst_range.size());
  for (size_t i = 0; i < src_range.size(); ++i) {
    ICHECK(analyzer->CanProveEqual(src_range[i]->extent, dst_range[i]->extent))
        << "multimem expects matching source and destination extents at "
           "dimension "
        << i << ", got " << src_range[i]->extent << " and "
        << dst_range[i]->extent;
  }

  if (mode == MultimemMode::kTmaStore || mode == MultimemMode::kTmaRedStore) {
    return LowerBulkCopy(T, analyzer);
  }
  if (IsPacked16BitMultimem()) {
    return LowerPacked16Bit(T, analyzer);
  }

  Buffer local_buf = (mode == MultimemMode::kLdReduce) ? dst : src;
  Array<Range> local_range =
      (mode == MultimemMode::kLdReduce) ? dst_range : src_range;
  RequireRegionInBounds(local_buf, local_range, analyzer, "multimem local");

  // Step 1-2: Create SIMT loop and fuse/transform
  auto transformed_loop = MakeTransformedSIMTLoop(analyzer);

  // Step 3: Create ParallelOp and run InferLayout at multiple levels
  auto par_op = ParallelOp(transformed_loop);

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({T.target,
                         T.thread_bounds,
                         T.layout_map,
                         analyzer,
                         false,
                         T.buffer_remap,
                         {},
                         false},
                        level);
  }

  // Step 4: Lower the parallel loop (PartitionLoop + VectorizeLoop)
  auto loop_layout = par_op->GetLoopLayout();
  Stmt result =
      LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var, analyzer,
                        {}, par_op->GetPredicate(T.thread_var));

  // Step 5: Post-process — replace mcast buffer accesses with multimem
  // call_extern
  Buffer mcast_buf = (mode == MultimemMode::kLdReduce) ? src : dst;
  // Remap the mcast buffer if needed
  if (T.buffer_remap.count(mcast_buf)) {
    mcast_buf = T.buffer_remap[mcast_buf];
  }
  result = MultimemRewriter(mcast_buf, mode, reduce_op).Rewrite(result);
  return result;
}

For MultimemOpNode::MakeTransformedSIMTLoop(arith::Analyzer *analyzer) const {
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  return Downcast<For>(ParallelLoopTransformer::Substitute(fused_loop));
}

bool MultimemOpNode::IsPacked16BitMultimem() const {
  if (mode == MultimemMode::kTmaStore || mode == MultimemMode::kTmaRedStore) {
    return false;
  }
  return (src->dtype.is_float16() || src->dtype.is_bfloat16()) &&
         src->dtype == dst->dtype;
}

Stmt MultimemOpNode::LowerPacked16Bit(const LowerArgs &T,
                                      arith::Analyzer *analyzer) const {
  Buffer local_buf = (mode == MultimemMode::kLdReduce) ? dst : src;
  Buffer mcast_buf = (mode == MultimemMode::kLdReduce) ? src : dst;
  Array<Range> local_range =
      (mode == MultimemMode::kLdReduce) ? dst_range : src_range;
  Array<Range> mcast_range =
      (mode == MultimemMode::kLdReduce) ? src_range : dst_range;

  Optional<PrimExpr> mcast_in_bounds = ValidatePacked16BitRegions(
      local_buf, local_range, mcast_buf, mcast_range, analyzer);
  ICHECK(T.layout_map.count(local_buf))
      << "multimem packed x2 lowering requires an inferred local fragment "
         "layout";
  Fragment expected_layout =
      MakePacked16BitLayout(local_buf, local_range, T.thread_bounds, analyzer);
  RequirePacked16BitLayout(T.layout_map[local_buf], expected_layout, analyzer);
  const size_t ndim = local_range.size();

  const PrimExpr last_extent =
      analyzer->Simplify(local_range[ndim - 1]->extent);

  Buffer remapped_local = local_buf;
  Buffer remapped_mcast = mcast_buf;
  if (T.buffer_remap.count(remapped_local)) {
    remapped_local = T.buffer_remap[remapped_local];
  }
  if (T.buffer_remap.count(remapped_mcast)) {
    remapped_mcast = T.buffer_remap[remapped_mcast];
  }
  ValidatePacked16BitPhysicalBuffers(remapped_local, remapped_mcast,
                                     mcast_range, analyzer);

  PrimExpr numel = ProductExtent(local_range, 0, ndim);
  PrimExpr leading_elements = ProductExtent(local_range, 0, ndim - 1);
  PrimExpr pairs_per_row = analyzer->Simplify(floordiv(last_extent, 2));
  PrimExpr total_pairs = analyzer->Simplify(leading_elements * pairs_per_row);
  PrimExpr thread_extent = T.thread_bounds->extent;
  PrimExpr thread_offset = T.thread_var - T.thread_bounds->min;
  PrimExpr trip_count = analyzer->Simplify(
      floordiv(total_pairs + thread_extent - 1, thread_extent));

  Var loop_var("multimem_pair_iter", DataType::Int(32));
  PrimExpr pair_id =
      analyzer->Simplify(loop_var * thread_extent + thread_offset);
  PrimExpr linear_leading =
      analyzer->Simplify(floordiv(pair_id, pairs_per_row));
  PrimExpr last_pair =
      analyzer->Simplify(pair_id - linear_leading * pairs_per_row);
  PrimExpr local_offset = analyzer->Simplify(loop_var * 2);

  auto make_indices = [&](const Array<Range> &ranges,
                          const Buffer &buffer) -> Array<PrimExpr> {
    Array<PrimExpr> indices;
    PrimExpr remaining = linear_leading;
    for (size_t i = 0; i + 1 < ndim; ++i) {
      PrimExpr stride = ProductExtent(ranges, i + 1, ndim - 1);
      PrimExpr coord = analyzer->Simplify(floordiv(remaining, stride));
      remaining = analyzer->Simplify(remaining - coord * stride);
      indices.push_back(analyzer->Simplify(ranges[i]->min + coord));
    }
    indices.push_back(
        analyzer->Simplify(ranges[ndim - 1]->min + last_pair * 2));
    ICHECK_EQ(indices.size(), buffer->shape.size());
    return indices;
  };

  Array<PrimExpr> local_indices =
      UnflattenIndex(local_offset, remapped_local->shape, analyzer);
  Array<PrimExpr> mcast_indices = make_indices(mcast_range, remapped_mcast);

  Array<PrimExpr> args;
  args.push_back(StringImm(
      multimem_detail::FuncName(mode, reduce_op, 2, local_buf->dtype)));
  if (mode == MultimemMode::kLdReduce) {
    args.push_back(multimem_detail::MakeAddress(remapped_local, local_indices));
    args.push_back(multimem_detail::MakeAddress(remapped_mcast, mcast_indices));
  } else {
    args.push_back(multimem_detail::MakeAddress(remapped_mcast, mcast_indices));
    args.push_back(multimem_detail::MakeAddress(remapped_local, local_indices));
  }
  Stmt body = Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
  body = IfThenElse(pair_id < total_pairs, body);
  Stmt packed_loop =
      For(loop_var, 0, trip_count, ForKind::kSerial, std::move(body));
  if (!mcast_in_bounds.defined()) {
    return packed_loop;
  }

  if (mode != MultimemMode::kLdReduce) {
    return IfThenElse(mcast_in_bounds.value(), packed_loop);
  }

  Array<PrimExpr> next_local_indices =
      UnflattenIndex(local_offset + 1, remapped_local->shape, analyzer);
  Array<Stmt> zero_stores{
      BufferStore(remapped_local, make_zero(remapped_local->dtype),
                  local_indices),
      BufferStore(remapped_local, make_zero(remapped_local->dtype),
                  next_local_indices),
  };
  Stmt zero_body =
      IfThenElse(pair_id < total_pairs, SeqStmt(std::move(zero_stores)));
  Stmt zero_loop =
      For(loop_var, 0, trip_count, ForKind::kSerial, std::move(zero_body));
  return IfThenElse(mcast_in_bounds.value(), packed_loop, zero_loop);
}

// === LowerBulkCopy ===
// CTA-collective bulk async store from shared to multicast global.
// Emits multimem.cp.async.bulk or multimem.cp.reduce.async.bulk PTX after
// validating the instruction's range, contiguity, size, and alignment rules.
namespace {

std::vector<PrimExpr> GetBulkCopyPhysicalStrides(const Buffer &buffer) {
  std::vector<PrimExpr> strides(buffer->shape.size());
  if (!buffer->strides.empty()) {
    ICHECK_EQ(buffer->strides.size(), buffer->shape.size())
        << "multimem TMA buffers with explicit strides must provide one stride "
           "per dimension";
    for (size_t i = 0; i < buffer->strides.size(); ++i) {
      strides[i] = buffer->strides[i];
    }
    return strides;
  }

  PrimExpr stride = 1;
  for (int i = static_cast<int>(buffer->shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride = stride * buffer->shape[i];
  }
  return strides;
}

struct BulkRegion {
  PrimExpr address;
  PrimExpr elements;
  Optional<PrimExpr> in_bounds;
};

struct BulkTransfer {
  BulkRegion shared;
  BulkRegion multicast;
  PrimExpr size_bytes;
  PrimExpr issue_predicate;
  std::string helper;
};

BulkRegion AnalyzeBulkRegion(const Buffer &logical_buffer,
                             const Buffer &physical_buffer,
                             const Array<Range> &ranges,
                             arith::Analyzer *analyzer,
                             const char *description) {
  ICHECK_EQ(logical_buffer->dtype, physical_buffer->dtype)
      << description << " buffer remap must preserve dtype";
  ICHECK_EQ(ranges.size(), logical_buffer->shape.size())
      << description << " region rank must match buffer rank";
  ICHECK_EQ(logical_buffer->shape.size(), physical_buffer->shape.size())
      << description << " buffer remap must preserve rank";
  ICHECK(!ranges.empty()) << description
                          << " region must have at least one dimension";
  for (size_t i = 0; i < ranges.size(); ++i) {
    ICHECK(analyzer->CanProveEqual(logical_buffer->shape[i],
                                   physical_buffer->shape[i]))
        << description << " buffer remap must preserve shape at dimension "
        << i;
    ICHECK(analyzer->CanProve(ranges[i]->extent > 0,
                              arith::ProofStrength::kSymbolicBound))
        << description << " region extents must be provably positive, got "
        << ranges[i]->extent << " at dimension " << i;
  }

  Optional<PrimExpr> in_bounds = AnalyzeRegionBounds(
      logical_buffer, ranges, RegionBoundsPolicy::kAllowWholeTilePredicate,
      analyzer, description);
  std::vector<PrimExpr> strides = GetBulkCopyPhysicalStrides(physical_buffer);
  PrimExpr contiguous_stride = 1;
  for (int i = static_cast<int>(ranges.size()) - 1; i >= 0; --i) {
    PrimExpr extent = analyzer->Simplify(ranges[i]->extent);
    if (!analyzer->CanProveEqual(extent, 1)) {
      ICHECK(analyzer->CanProveEqual(strides[i], contiguous_stride))
          << description
          << " region must be provably physically contiguous; dimension " << i
          << " has physical stride " << strides[i]
          << " but a contiguous region requires stride " << contiguous_stride;
    }
    contiguous_stride = analyzer->Simplify(contiguous_stride * extent);
  }

  Array<PrimExpr> start_indices;
  for (const Range &range : ranges) {
    start_indices.push_back(analyzer->Simplify(range->min));
  }
  Array<PrimExpr> element_offsets = physical_buffer->ElemOffset(start_indices);
  ICHECK_EQ(element_offsets.size(), 1)
      << description
      << " region must map to one contiguous physical address range";
  int element_bytes =
      physical_buffer->dtype.bytes() * physical_buffer->dtype.lanes();
  PrimExpr byte_offset = analyzer->Simplify(element_offsets[0] * element_bytes);
  RequireBufferBaseAlignment(physical_buffer, 16, description);
  ICHECK(analyzer->CanProveEqual(FloorMod(byte_offset, 16), 0))
      << description
      << " start address must be provably 16-byte aligned, got byte offset "
      << byte_offset;

  PrimExpr elements = make_const(DataType::Int(64), 1);
  for (const Range &range : ranges) {
    elements *= Cast(DataType::Int(64), range->extent);
  }
  return {multimem_detail::MakeAddress(physical_buffer, start_indices),
          analyzer->Simplify(elements), in_bounds};
}

std::string GetBulkCopyReduceFuncName(int reduce_op, DataType dtype) {
  bool is_f32 = dtype.is_float() && dtype.bits() == 32 && dtype.lanes() == 1;
  bool is_f16 = dtype.is_float16() && dtype.lanes() == 1;
  bool is_bf16 = dtype.is_bfloat16() && dtype.lanes() == 1;
  ICHECK(is_f32 || is_f16 || is_bf16)
      << "multimem TMA reduction supports float32, float16, and bfloat16, got "
      << dtype;

  std::string dtype_suffix = is_f32 ? "f32" : (is_f16 ? "f16" : "bf16");
  std::string op_name;
  switch (reduce_op) {
  case 0:
    op_name = "add";
    break;
  case 1:
    ICHECK(!is_f32)
        << "multimem TMA reduction does not support MIN with float32";
    op_name = "min";
    break;
  case 2:
    ICHECK(!is_f32)
        << "multimem TMA reduction does not support MAX with float32";
    op_name = "max";
    break;
  default:
    LOG(FATAL) << "Invalid multimem TMA reduce_op: " << reduce_op;
  }
  return "tl::multimem::cp_reduce_async_bulk_" + op_name + "_" + dtype_suffix;
}

Buffer ResolveBulkBuffer(const Buffer &buffer, const LowerArgs &T,
                         const char *description) {
  ICHECK(!T.layout_map.count(buffer))
      << description
      << " does not support layout-remapped buffers because the bulk "
         "instruction copies a physically contiguous byte range";
  if (T.buffer_remap.count(buffer)) {
    Buffer remapped = T.buffer_remap[buffer];
    ICHECK(!T.layout_map.count(remapped))
        << description
        << " does not support layout-remapped buffers because the bulk "
           "instruction copies a physically contiguous byte range";
    return remapped;
  }
  return buffer;
}

BulkTransfer AnalyzeBulkTransfer(const Buffer &shared,
                                 const Array<Range> &shared_ranges,
                                 const Buffer &multicast,
                                 const Array<Range> &multicast_ranges,
                                 MultimemMode mode, int reduce_op,
                                 const LowerArgs &T,
                                 arith::Analyzer *analyzer) {
  ICHECK_EQ(shared->dtype.bits() % 8, 0)
      << "multimem TMA requires byte-addressable element dtypes, got "
      << shared->dtype;

  Buffer physical_shared =
      ResolveBulkBuffer(shared, T, "multimem TMA shared source");
  Buffer physical_multicast =
      ResolveBulkBuffer(multicast, T, "multimem TMA multicast destination");
  BulkRegion shared_region = AnalyzeBulkRegion(
      shared, physical_shared, shared_ranges, analyzer, "multimem TMA shared");
  BulkRegion multicast_region =
      AnalyzeBulkRegion(multicast, physical_multicast, multicast_ranges,
                        analyzer, "multimem TMA multicast");

  int element_bytes = shared->dtype.bytes() * shared->dtype.lanes();
  PrimExpr size_bytes =
      analyzer->Simplify(shared_region.elements * element_bytes);
  ICHECK(
      analyzer->CanProve(size_bytes > 0, arith::ProofStrength::kSymbolicBound))
      << "multimem TMA transfer size must be provably positive, got "
      << size_bytes;
  ICHECK(analyzer->CanProveEqual(
      FloorMod(size_bytes, make_const(size_bytes.dtype(), 16)), 0))
      << "multimem TMA transfer size must be provably divisible by 16 bytes, "
         "got "
      << size_bytes;
  arith::ConstIntBound size_bound = analyzer->const_int_bound(size_bytes);
  ICHECK(size_bound->max_value != arith::ConstIntBound::kPosInf &&
         size_bound->max_value <=
             static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "multimem TMA transfer size must fit in a uint32 byte count, got "
      << size_bytes;

  PrimExpr issue_predicate = EQ(T.thread_var, T.thread_bounds->min);
  if (shared_region.in_bounds.defined()) {
    issue_predicate = And(issue_predicate, shared_region.in_bounds.value());
  }
  if (multicast_region.in_bounds.defined()) {
    issue_predicate = And(issue_predicate, multicast_region.in_bounds.value());
  }

  std::string helper = mode == MultimemMode::kTmaRedStore
                           ? GetBulkCopyReduceFuncName(reduce_op, shared->dtype)
                           : "tl::multimem::cp_async_bulk";
  return {shared_region, multicast_region, size_bytes,
          analyzer->Simplify(issue_predicate), helper};
}

} // namespace

Stmt MultimemOpNode::LowerBulkCopy(const LowerArgs &T,
                                   arith::Analyzer *analyzer) const {
  BulkTransfer transfer = AnalyzeBulkTransfer(src, src_range, dst, dst_range,
                                              mode, reduce_op, T, analyzer);

  Array<PrimExpr> extern_args;
  extern_args.push_back(StringImm(transfer.helper));
  extern_args.push_back(transfer.multicast.address);
  extern_args.push_back(transfer.shared.address);
  extern_args.push_back(Cast(DataType::UInt(32), transfer.size_bytes));

  Stmt bulk_copy =
      Evaluate(Call(DataType::Handle(), builtin::call_extern(), extern_args));
  return IfThenElse(transfer.issue_predicate, bulk_copy);
}

// === Clone ===
TileOperator MultimemOpNode::Clone() const {
  auto node = tvm::ffi::make_object<MultimemOpNode>(*this);
  return MultimemOp(node);
}

// === Registration ===
TIR_REGISTER_TL_TILE_OP(MultimemOp, multimem)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
