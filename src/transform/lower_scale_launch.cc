/*!
 * \file lower_scale_launch.cc
 * \brief Lower frontend T.Scale launch axes to existing CUDA launch bindings.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/distributed.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

constexpr const char *kScale = "tl.scale";
constexpr const char *kScaleName = "tl.scale.name";
constexpr const char *kScaleBind = "tl.scale.bind";
constexpr const char *kScaleNumSmsPerDie = "tl.scale.num_sms_per_die";
constexpr const char *kScaleClusterSize = "tl.scale.cluster_size";
constexpr const char *kScaleSmSchedule = "tl.scale.sm_schedule";
constexpr const char *kScaleThreadVar = "tl.scale.thread_var";
constexpr const char *kScaleSwizzle = "tl.scale.swizzle";
constexpr const char *kScaleSwizzleOrder = "tl.scale.swizzle_order";
constexpr const char *kScaleWorkgroup = "tl.scale.workgroup";
constexpr const char *kScaleIsClusterRank = "tl.scale.is_cluster_rank";
constexpr int kWarpSize = 32;

struct ScaleAxis {
  Var var;
  PrimExpr extent;
  std::string name;
  std::string bind;
  Map<String, ffi::Any> annotations;
};

bool IsScaleLoop(const ForNode *op) {
  if (op->kind != ForKind::kSerial) {
    return false;
  }
  auto marker = op->annotations.Get(kScale);
  if (!marker) {
    return false;
  }
  if (auto b = marker.value().try_cast<Bool>()) {
    return b.value()->value;
  }
  return true;
}

// True if `body` contains a scale loop anywhere in its subtree.
bool ContainsScaleLoop(const Stmt &body) {
  bool found = false;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (found) return;
    if (const auto *f = node.as<ForNode>()) {
      if (IsScaleLoop(f)) found = true;
    }
  });
  return found;
}

PrimExpr GetPrimExprAnnotation(const ScaleAxis &axis, const char *key) {
  if (auto value = axis.annotations.Get(key)) {
    if (auto expr = value.value().try_cast<PrimExpr>()) {
      return expr.value();
    }
    LOG(FATAL) << "`" << key << "` must be a PrimExpr-compatible value";
  }
  return PrimExpr();
}

// Read the per-dimension workgroup shape (``tl.scale.workgroup``), padded to 3
// dims with 1s. Returns {1,1,1} when the annotation is absent.
std::array<int64_t, 3> GetWorkgroupDims(const ScaleAxis &axis) {
  std::array<int64_t, 3> dims = {1, 1, 1};
  auto wg = axis.annotations.Get(kScaleWorkgroup);
  if (!wg.has_value()) {
    return dims;
  }
  if (auto arr = wg.value().try_cast<Array<ffi::Any>>()) {
    auto vals = arr.value();
    ICHECK_LE(vals.size(), 3U)
        << "T.Scale workgroup supports at most 3 dimensions";
    for (size_t i = 0; i < vals.size(); ++i) {
      if (auto e = vals[i].try_cast<PrimExpr>()) {
        const auto *imm = e.value().as<IntImmNode>();
        ICHECK(imm != nullptr)
            << "T.Scale block workgroup dims must be constant integers";
        dims[i] = imm->value;
      }
    }
  }
  return dims;
}

std::string GetStringAnnotation(const Map<String, ffi::Any> &annotations,
                                const char *key,
                                std::string default_value = "") {
  if (auto value = annotations.Get(key)) {
    if (auto str = value.value().try_cast<String>()) {
      return static_cast<std::string>(str.value());
    }
    if (auto str_imm = value.value().try_cast<StringImm>()) {
      return str_imm.value()->value;
    }
    LOG(FATAL) << "`" << key << "` must be a string";
  }
  return default_value;
}

bool IsOne(PrimExpr expr) {
  arith::Analyzer analyzer;
  return is_one(analyzer.Simplify(std::move(expr)));
}

IterVar MakeThreadIterVar(const Var &var, PrimExpr extent,
                          const std::string &thread_tag) {
  return IterVar(Range::FromMinExtent(Integer(0), std::move(extent)), var,
                 IterVarType::kThreadIndex, String(thread_tag));
}

Call MakeBlockRankInCluster() {
  return Call(DataType::Int(32), tl::block_rank_in_cluster(), {});
}

Call MakeGetSmid() { return Call(DataType::Int(32), tl::get_smid(), {}); }

PrimExpr MaybeCast(PrimExpr expr, const DataType &dtype) {
  if (expr.dtype() == dtype) {
    return expr;
  }
  return Cast(dtype, std::move(expr));
}

// ---------------------------------------------------------------------------
// Scope tree.
//
// A lexical tree of the scale `For` loops in a PrimFunc body. Unlike the old
// chain collection, the builder descends through ALL statement structure
// (SeqStmt / IfThenElse / serial loops), so a scale hidden behind non-scale
// statements is still linked to its lexical parent scale. The tree is the single
// analysis input to launch-group assembly, validation, substitution, and the
// rewrite.
// ---------------------------------------------------------------------------

std::array<int64_t, 3>
GetWorkgroupDimsFromAnnotations(const Map<String, ffi::Any> &annotations) {
  std::array<int64_t, 3> dims = {1, 1, 1};
  auto wg = annotations.Get(kScaleWorkgroup);
  if (!wg.has_value()) {
    return dims;
  }
  if (auto arr = wg.value().try_cast<Array<ffi::Any>>()) {
    auto vals = arr.value();
    if (vals.size() > 3U) {
      return dims;
    }
    for (size_t i = 0; i < vals.size(); ++i) {
      if (auto e = vals[i].try_cast<PrimExpr>()) {
        if (const auto *imm = e.value().as<IntImmNode>()) {
          dims[i] = imm->value;
        }
      }
    }
  }
  return dims;
}

enum class ScaleKind {
  kClusterGrid,
  kClusterRankBlock,
  kLogicalBlock,
  kThread,
  kWarp,
  kPhysSm,
  kPhysDie,
  kPhysClusterRank,
  kDevice,
  kUnknown,
};

struct ScaleNode {
  const ForNode *loop = nullptr;  // original scale For (identity preserved)
  Var var;                        // rank / loop var
  PrimExpr extent;                // loop extent
  std::string bind;               // resolved bind tag (authoritative)
  Map<String, ffi::Any> annotations;
  std::array<int64_t, 3> workgroup = {1, 1, 1};  // cluster_dims source
  ScaleNode *parent = nullptr;
  std::vector<ScaleNode *> children;
  bool under_if = false;  // true if this scale is inside an IfThenElse subtree
  ScaleKind kind = ScaleKind::kUnknown;
};

// Builds the lexical scale tree. Tracks whether the current position is inside
// an IfThenElse so a scale nested in a branch can be rejected loudly.
class ScaleTreeBuilder : public StmtVisitor {
public:
  std::vector<std::unique_ptr<ScaleNode>> storage;  // owns all nodes
  std::vector<ScaleNode *> roots;  // outermost scales in the scanned subtree

  void Build(const Stmt &s) { VisitStmt(s); }

private:
  std::vector<ScaleNode *> stack_;
  bool under_if_ = false;

  void VisitStmt_(const IfThenElseNode *op) final {
    bool saved = under_if_;
    under_if_ = true;
    StmtVisitor::VisitStmt_(op);
    under_if_ = saved;
  }

  void VisitStmt_(const ForNode *op) final {
    if (!IsScaleLoop(op)) {
      StmtVisitor::VisitStmt_(op);
      return;
    }
    auto node = std::make_unique<ScaleNode>();
    node->loop = op;
    node->var = op->loop_var;
    node->extent = op->extent;
    node->annotations = op->annotations;
    std::string name = GetStringAnnotation(op->annotations, kScaleName);
    node->bind = GetStringAnnotation(op->annotations, kScaleBind, "");
    if (node->bind.empty()) {
      node->bind = name;
    }
    node->workgroup = GetWorkgroupDimsFromAnnotations(op->annotations);
    node->under_if = under_if_;
    ScaleNode *raw = node.get();
    if (!stack_.empty()) {
      raw->parent = stack_.back();
      stack_.back()->children.push_back(raw);
    } else {
      roots.push_back(raw);
    }
    storage.push_back(std::move(node));
    stack_.push_back(raw);
    // The scale body opens a fresh nesting context; an `if` directly enclosing
    // *this* scale was recorded via under_if above, but the scale's own body
    // starts clean.
    bool saved = under_if_;
    under_if_ = false;
    StmtVisitor::VisitStmt_(op);
    under_if_ = saved;
    stack_.pop_back();
  }
};

void ClassifyScaleTree(ScaleNode *node, bool cluster_ancestor) {
  const std::string &b = node->bind;
  bool cluster_here = cluster_ancestor;
  if (b == "cluster") {
    node->kind = ScaleKind::kClusterGrid;
    cluster_here = true;
  } else if (b == "thread" || b == "threadIdx.x") {
    node->kind = ScaleKind::kThread;
  } else if (b == "warp") {
    node->kind = ScaleKind::kWarp;
  } else if (b == "sm" || b == "physical_sm" || b == "physical_sm_cluster") {
    node->kind = ScaleKind::kPhysSm;
  } else if (b == "die" || b == "physical_die") {
    node->kind = ScaleKind::kPhysDie;
  } else if (b == "cta" || b == "sm-cluster" || b == "cluster_rank") {
    node->kind = ScaleKind::kPhysClusterRank;
  } else if (b == "block" || b == "blockIdx.x" || b == "logical") {
    bool is_cluster_rank = cluster_ancestor;
    if (auto v = node->annotations.Get(kScaleIsClusterRank)) {
      if (auto bb = v.value().try_cast<Bool>()) {
        is_cluster_rank = bb.value()->value;
      }
    }
    node->kind = is_cluster_rank ? ScaleKind::kClusterRankBlock
                                 : ScaleKind::kLogicalBlock;
  } else if (b == "device") {
    node->kind = ScaleKind::kDevice;
  } else {
    node->kind = ScaleKind::kUnknown;
  }
  for (auto *c : node->children) {
    ClassifyScaleTree(c, cluster_here);
  }
}

// Replaces every scale `For` node in `targets` (matched by pointer identity)
// with its body, leaving all surrounding structure (SeqStmt / serial / T.assume)
// in place and in order. This is the unified scope-tree rewriter: it splices out
// a launch group's scale loops without moving any user statement, so the
// remaining body can be lowered and wrapped by the launch bindings.
class SpliceOutScales : public StmtMutator {
public:
  explicit SpliceOutScales(std::unordered_set<const ForNode *> targets)
      : targets_(std::move(targets)) {}

private:
  std::unordered_set<const ForNode *> targets_;
  Stmt VisitStmt_(const ForNode *op) final {
    if (targets_.count(op)) {
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }
};

// True if `s` is a pure no-op statement (T.assume / Evaluate(const)) with no
// scale loop inside.
bool StmtIsNoOp(const Stmt &s) {
  if (const auto *ev = s.as<EvaluateNode>()) {
    if (const auto *call = ev->value.as<CallNode>()) {
      if (call->op.same_as(builtin::assume())) return true;
    }
    if (ev->value.as<IntImmNode>()) return true;
  }
  return false;
}

// True if the path from `node` down to `child` (the child scale For) passes
// through ONLY no-op statements: SeqStmt, pass-through serial loops, T.assume,
// and Evaluate(const). Does NOT look inside `child`'s own body. Used to validate
// that statements interleaved between two scale levels are safe to leave in
// place under the launch bindings.
bool IsNoOpPathToChild(const Stmt &node, const ForNode *child) {
  if (const auto *f = node.as<ForNode>()) {
    if (f == child) return true;  // reached the child scale: path is clean
    if (IsScaleLoop(f)) return false;  // a different scale on the path
    if (f->kind != ForKind::kSerial) return false;  // parallel/unroll/vectorize
    return IsNoOpPathToChild(f->body, child);
  }
  if (const auto *seq = node.as<SeqStmtNode>()) {
    for (const auto &s : seq->seq) {
      if (ContainsScaleLoop(s)) {
        // The element that contains the child must itself be a clean path.
        if (!IsNoOpPathToChild(s, child)) return false;
      } else if (!StmtIsNoOp(s)) {
        // Any other sibling must be a pure no-op (no side effect).
        return false;
      }
    }
    return true;
  }
  return StmtIsNoOp(node);
}

} // namespace

namespace {

// A launch group is the set of scale nodes that together decide one launch
// configuration (grid + cluster + threads). The current implementation supports
// exactly one launch group per PrimFunc.
struct LaunchGroup {
  std::vector<ScaleNode *> ordered;  // all member scales, outer -> inner
  const ForNode *anchor = nullptr;   // outermost scale loop; launch attrs wrap
                                     // its (rewritten) body in place
};

class ScaleLaunchLowerer : public StmtMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    Map<Var, Buffer> buffer_data_to_buffer;
    for (const auto &kv : f->buffer_map) {
      buffer_data_to_buffer.Set(kv.first, kv.second);
      buffer_data_to_buffer.Set(kv.second->data, kv.second);
    }

    // Build + classify the lexical scale tree.
    ScaleTreeBuilder builder;
    builder.Build(f->body);
    if (builder.roots.empty()) {
      return f;  // no scale loops: nothing to do
    }
    for (auto *r : builder.roots) {
      ClassifyScaleTree(r, /*cluster_ancestor=*/false);
    }

    // Assemble the (single) launch group from the tree spine and validate it.
    LaunchGroup group = AssembleLaunchGroup(builder.roots);
    ValidateScaleTree(builder.roots, group);

    ScaleLaunchLowerer lowerer(std::move(buffer_data_to_buffer), &group);
    Stmt body = lowerer.VisitStmt(f->body);
    f.CopyOnWrite()->body = body;
    if (lowerer.cluster_dims_.defined()) {
      f = WithAttr(std::move(f), "cluster_dims",
                   lowerer.cluster_dims_.value());
    }
    return f;
  }

private:
  using Parent = StmtMutator;

  ScaleLaunchLowerer(Map<Var, Buffer> buffer_data_to_buffer,
                     const LaunchGroup *group)
      : group_(group),
        buffer_data_to_buffer_(std::move(buffer_data_to_buffer)) {}

  // Collect the launch group: the spine of scales reachable from the single
  // root, descending into the unique scale child at each level. The members are
  // collected in lexical outer->inner order; the anchor is the outermost.
  static LaunchGroup AssembleLaunchGroup(
      const std::vector<ScaleNode *> &roots) {
    ICHECK_EQ(roots.size(), 1U)
        << "T.scale: multiple independent launch groups (parallel top-level "
           "scales) are not supported yet.";
    LaunchGroup group;
    ScaleNode *cur = roots[0];
    group.anchor = cur->loop;
    while (cur != nullptr) {
      group.ordered.push_back(cur);
      if (cur->children.empty()) {
        cur = nullptr;
      } else {
        ICHECK_EQ(cur->children.size(), 1U)
            << "T.scale: multiple independent launch groups (sibling scales "
               "under one scale) are not supported yet.";
        cur = cur->children[0];
      }
    }
    return group;
  }

  // Loud-error validation, centralized. Runs on the classified tree + group.
  static void ValidateScaleTree(const std::vector<ScaleNode *> &roots,
                                const LaunchGroup &group) {
    // 1. No scale may sit inside an IfThenElse branch.
    for (const auto &r : roots) {
      ValidateNoScaleUnderIf(r);
    }

    // 2. Edge checks along the group spine: every parent->child edge must be a
    // direct nesting OR (for the non-physical case) a no-op-only path. Physical
    // scales (sm/die/sm-cluster) must stay strictly direct-chained.
    bool group_has_physical = false;
    int cluster_rank_block_count = 0;
    int thread_warp_count = 0;
    bool seen_cluster = false;
    for (size_t i = 0; i < group.ordered.size(); ++i) {
      ScaleNode *n = group.ordered[i];
      switch (n->kind) {
      case ScaleKind::kPhysSm:
      case ScaleKind::kPhysDie:
      case ScaleKind::kPhysClusterRank:
        group_has_physical = true;
        break;
      case ScaleKind::kClusterGrid:
        seen_cluster = true;
        break;
      case ScaleKind::kClusterRankBlock:
        ++cluster_rank_block_count;
        break;
      case ScaleKind::kThread:
      case ScaleKind::kWarp:
        ++thread_warp_count;
        break;
      default:
        break;
      }
    }

    // block -> cluster reverse order: a cluster grid whose lexical ancestor is a
    // block is illegal hierarchy.
    for (size_t i = 0; i < group.ordered.size(); ++i) {
      if (group.ordered[i]->kind == ScaleKind::kClusterGrid) {
        for (size_t j = 0; j < i; ++j) {
          ScaleKind k = group.ordered[j]->kind;
          ICHECK(k != ScaleKind::kLogicalBlock &&
                 k != ScaleKind::kClusterRankBlock)
              << "T.scale: a `block` scale before a `cluster` scale is not "
                 "supported yet. The legal hierarchy is `cluster -> block -> "
                 "thread`; `block -> cluster` is invalid.";
        }
      }
    }

    // cluster-rank block must have a cluster ancestor in the group.
    if (cluster_rank_block_count > 0) {
      ICHECK(seen_cluster)
          << "T.scale: a cluster-internal `block` scale must be nested under a "
             "`cluster` scale.";
    }

    // At most one thread/warp scale.
    ICHECK_LE(thread_warp_count, 1)
        << "T.scale: more than one thread/warp scale in a launch group is not "
           "supported yet.";

    // A thread/warp scale must be terminal: it must be the innermost scale in
    // the launch group. Any scale (block / cluster / sm / die / thread / warp /
    // device) nested below a thread/warp would otherwise be silently folded into
    // the axes vector and mis-lowered (e.g. a block below a thread becomes a
    // spurious extra grid dimension).
    for (size_t i = 0; i < group.ordered.size(); ++i) {
      ScaleKind k = group.ordered[i]->kind;
      if (k == ScaleKind::kThread || k == ScaleKind::kWarp) {
        ICHECK_EQ(i, group.ordered.size() - 1)
            << "T.scale: a thread/warp scale must be the terminal (innermost) "
               "scale; nesting another scale below a thread/warp scale is not "
               "supported yet.";
      }
    }

    // Edge validation: each parent->child edge must be a clean path. Physical
    // groups require strict direct nesting; logical groups allow no-op paths.
    for (size_t i = 0; i + 1 < group.ordered.size(); ++i) {
      ScaleNode *parent = group.ordered[i];
      ScaleNode *child = group.ordered[i + 1];
      const ForNode *direct = parent->loop->body.as<ForNode>();
      bool is_direct = (direct == child->loop);
      if (group_has_physical) {
        ICHECK(is_direct)
            << "T.scale: physical launch scales (die / sm / sm-cluster / "
               "thread) must be directly nested; statements between them are "
               "not supported yet.";
      } else {
        ICHECK(is_direct || IsNoOpPathToChild(parent->loop->body, child->loop))
            << "T.scale: only no-op statements (T.assume) and pass-through "
               "T.serial loops are allowed between scale levels. Stores, "
               "T.copy/T.alloc, and `if` between scale levels are not supported "
               "yet; place such statements inside the innermost scale.";
      }
    }
  }

  static void ValidateNoScaleUnderIf(ScaleNode *node) {
    ICHECK(!node->under_if)
        << "T.scale: a scale inside an `if` branch is not supported yet. Move "
           "the scale outside the `if`, or move the condition inside the "
           "innermost scale.";
    for (auto *c : node->children) {
      ValidateNoScaleUnderIf(c);
    }
  }

  // Rewrite: when we hit the launch group's anchor For, splice out all the
  // group's scale loops (keeping interleaved statements in place), lower the
  // axes, and wrap the launch attrs around the anchor's rewritten body. The
  // anchor For is replaced in place, so statements outside the anchor stay where
  // they are.
  Stmt VisitStmt_(const ForNode *op) final {
    if (op != group_->anchor) {
      return Parent::VisitStmt_(op);
    }

    // Build the axes vector (lexical outer -> inner) and the set of scale loops
    // to splice out of the anchor subtree.
    std::vector<ScaleAxis> axes;
    std::unordered_set<const ForNode *> scale_loops;
    for (ScaleNode *n : group_->ordered) {
      ScaleAxis axis;
      axis.var = n->var;
      axis.extent = n->extent;
      axis.annotations = n->annotations;
      axis.name = GetStringAnnotation(n->annotations, kScaleName);
      axis.bind = n->bind;
      axes.push_back(std::move(axis));
      scale_loops.insert(n->loop);
    }

    // Splice every scale For out of the anchor subtree, leaving interleaved
    // statements (SeqStmt / serial / T.assume) in place and in order. The
    // anchor itself is included in `scale_loops`, so we start from its body.
    Stmt body = SpliceOutScales(scale_loops)(ffi::GetRef<Stmt>(op));

    return LowerAxes(axes, body);
  }

  Stmt LowerAxes(const std::vector<ScaleAxis> &axes, Stmt body) {
    std::optional<ScaleAxis> thread_axis;
    std::optional<ScaleAxis> cluster_rank_axis;
    std::optional<ScaleAxis> block_rank_axis;
    std::optional<ScaleAxis> die_axis;
    std::optional<ScaleAxis> sm_axis;
    std::vector<ScaleAxis> logical_axes;
    std::vector<ScaleAxis> cluster_grid_axes;
    PrimExpr grid_extent = Integer(1);
    Map<Var, PrimExpr> subst;
    std::optional<int64_t> swizzle_panel;
    std::string swizzle_order = "rasterization2DRow";

    bool has_cluster_grid = false;
    for (const auto &axis : axes) {
      if (axis.bind == "cluster") {
        has_cluster_grid = true;
        break;
      }
    }

    for (const auto &axis : axes) {
      auto sw = axis.annotations.Get(kScaleSwizzle);
      if (sw.has_value()) {
        if (auto imm = sw.value().try_cast<PrimExpr>()) {
          if (const auto *i = imm.value().as<IntImmNode>()) {
            swizzle_panel = i->value;
          }
        }
        std::string order =
            GetStringAnnotation(axis.annotations, kScaleSwizzleOrder, "row");
        swizzle_order = (order == "col" || order == "column")
                            ? "rasterization2DColumn"
                            : "rasterization2DRow";
      }
    }

    for (const auto &axis : axes) {
      std::string bind = axis.bind;
      bool counts_to_grid = true;
      if (bind == "thread" || bind == "threadIdx.x") {
        thread_axis = axis;
        counts_to_grid = false;
      } else if (bind == "sm-cluster" || bind == "cta" ||
                 bind == "cluster_rank") {
        cluster_rank_axis = axis;
      } else if (bind == "cluster") {
        cluster_grid_axes.push_back(axis);
      } else if (bind == "die" || bind == "physical_die") {
        die_axis = axis;
      } else if (bind == "sm" || bind == "physical_sm" ||
                 bind == "physical_sm_cluster") {
        sm_axis = axis;
      } else if (bind == "block" || bind == "blockIdx.x" ||
                 bind == "logical") {
        bool is_cluster_rank = has_cluster_grid;
        if (auto v = axis.annotations.Get(kScaleIsClusterRank)) {
          if (auto b = v.value().try_cast<Bool>()) {
            is_cluster_rank = b.value()->value;
          }
        }
        if (is_cluster_rank) {
          block_rank_axis = axis;
        } else {
          logical_axes.push_back(axis);
        }
      } else if (bind == "device") {
        // Single-device scopes keep the historical constant-0 rank. A
        // workgroup > 1 is SPMD across the process group: every rank runs the
        // body once with its own rank id (tl::get_rank() reads the peer table
        // installed by kernel.initialize(allocator)).
        const auto *ext = axis.extent.as<IntImmNode>();
        if (ext != nullptr && ext->value == 1) {
          subst.Set(axis.var, MaybeCast(Integer(0), axis.var.dtype()));
        } else {
          subst.Set(axis.var,
                    Call(axis.var.dtype(), tl::get_rank(), {}));
        }
        counts_to_grid = false;
      } else if (bind == "warp") {
        arith::Analyzer warp_analyzer;
        Var tv;
        auto tv_anno = axis.annotations.Get(kScaleThreadVar);
        if (tv_anno.has_value()) {
          auto maybe_expr = tv_anno.value().try_cast<PrimExpr>();
          if (maybe_expr.has_value()) {
            const auto *vn = maybe_expr.value().as<VarNode>();
            if (vn) tv = ffi::GetRef<Var>(vn);
          }
        }
        if (!tv.defined()) {
          tv = Var("v", DataType::Int(32));
        }
        ScaleAxis synth_thread;
        synth_thread.var = tv;
        synth_thread.extent = warp_analyzer.Simplify(axis.extent * kWarpSize);
        thread_axis = synth_thread;
        subst.Set(axis.var,
                  MaybeCast(FloorDiv(tv, Integer(kWarpSize)),
                            axis.var.dtype()));
        counts_to_grid = false;
      } else {
        LOG(FATAL) << "Unsupported T.Scale bind `" << bind
                   << "` for scale `" << axis.name << "`";
      }
      if (counts_to_grid) {
        grid_extent = grid_extent * axis.extent;
      }
    }

    if (block_rank_axis.has_value() && cluster_grid_axes.empty()) {
      ICHECK(false)
          << "T.scale: a cluster-internal `block` scale must be nested under "
             "its `cluster` scale.";
    }

    if (IsOne(grid_extent)) {
      grid_extent = Integer(1);
    }

    Var bx("bx", DataType::Int(32));         // physical path: linearized blockIdx.x
    std::vector<IterVar> block_ivars;        // logical path: native blockIdx.{x,y,z}

    bool physical_path = sm_axis.has_value();

    const std::vector<ScaleAxis> &grid_axes =
        has_cluster_grid ? cluster_grid_axes : logical_axes;

    std::array<int64_t, 3> blk_dims = {1, 1, 1};
    if (block_rank_axis.has_value()) {
      blk_dims = GetWorkgroupDims(*block_rank_axis);
      if (blk_dims[0] == 1 && blk_dims[1] == 1 && blk_dims[2] == 1) {
        const auto *e = block_rank_axis->extent.as<IntImmNode>();
        ICHECK(e != nullptr)
            << "T.Scale block rank extent must be a constant integer";
        blk_dims[0] = e->value;
      }
    }

    if (physical_path) {
      PrimExpr remaining = bx;
      if (block_rank_axis.has_value()) {
        subst.Set(block_rank_axis->var,
                  MaybeCast(MakeBlockRankInCluster(), block_rank_axis->var.dtype()));
        SetClusterDims(Array<Integer>{Integer(blk_dims[0]), Integer(blk_dims[1]),
                                      Integer(blk_dims[2])});
        remaining = FloorDiv(remaining, block_rank_axis->extent);
      }
      for (const auto &axis : grid_axes) {
        PrimExpr value = FloorMod(remaining, axis.extent);
        subst.Set(axis.var, MaybeCast(value, axis.var.dtype()));
        remaining = FloorDiv(remaining, axis.extent);
      }
      subst.Set(sm_axis->var, MaybeCast(remaining, sm_axis->var.dtype()));
    } else if (!grid_axes.empty()) {
      static const char *kBlockTags[] = {"blockIdx.x", "blockIdx.y",
                                         "blockIdx.z"};
      ICHECK_LE(grid_axes.size(), 3U)
          << "T.Scale supports at most 3 logical grid dimensions";
      if (block_rank_axis.has_value()) {
        subst.Set(block_rank_axis->var,
                  MaybeCast(MakeBlockRankInCluster(),
                            block_rank_axis->var.dtype()));
        SetClusterDims(Array<Integer>{Integer(blk_dims[0]), Integer(blk_dims[1]),
                                      Integer(blk_dims[2])});
      }
      for (size_t d = 0; d < grid_axes.size(); ++d) {
        const auto &axis = grid_axes[d];
        int64_t blk = blk_dims[d];
        arith::Analyzer ana;
        PrimExpr hw_ext = ana.Simplify(axis.extent * Integer(blk));
        Var bidx("bx" + std::to_string(d), DataType::Int(32));
        block_ivars.push_back(MakeThreadIterVar(bidx, hw_ext, kBlockTags[d]));
        PrimExpr coord =
            (blk == 1) ? PrimExpr(bidx) : FloorDiv(bidx, Integer(blk));
        subst.Set(axis.var, MaybeCast(coord, axis.var.dtype()));
      }
    }

    if (cluster_rank_axis.has_value()) {
      PrimExpr cluster_rank = MakeBlockRankInCluster();
      subst.Set(cluster_rank_axis->var,
                MaybeCast(cluster_rank, cluster_rank_axis->var.dtype()));
      SetClusterDims(*cluster_rank_axis);
    }

    bool needs_physical_sm =
        die_axis.has_value() || (sm_axis.has_value() &&
                                 (sm_axis->bind == "sm" ||
                                  sm_axis->bind == "physical_sm" ||
                                  sm_axis->bind == "physical_sm_cluster"));
    if (needs_physical_sm) {
      ICHECK(sm_axis.has_value())
          << "T.Scale physical die requires a companion physical sm axis";
      PrimExpr num_sms_per_die;
      if (die_axis.has_value()) {
        num_sms_per_die =
            GetPrimExprAnnotation(*die_axis, kScaleNumSmsPerDie);
      }
      if (!num_sms_per_die.defined() && sm_axis.has_value()) {
        num_sms_per_die = GetPrimExprAnnotation(*sm_axis, kScaleNumSmsPerDie);
      }
      PrimExpr cluster_size = GetPrimExprAnnotation(*sm_axis, kScaleClusterSize);
      if (!cluster_size.defined() && cluster_rank_axis.has_value()) {
        cluster_size =
            GetPrimExprAnnotation(*cluster_rank_axis, kScaleClusterSize);
      }
      if (!cluster_size.defined()) {
        cluster_size = cluster_rank_axis.has_value()
                           ? cluster_rank_axis->extent
                           : PrimExpr(Integer(1));
      }
      if (!num_sms_per_die.defined()) {
        num_sms_per_die = sm_axis->extent * cluster_size;
      }

      PrimExpr logical_sm = MakeLogicalSm(*sm_axis);
      PrimExpr die = FloorDiv(logical_sm, num_sms_per_die);
      PrimExpr local_sm = logical_sm - die * num_sms_per_die;
      PrimExpr local_cluster = FloorDiv(local_sm, cluster_size);
      if (die_axis.has_value()) {
        subst.Set(die_axis->var, MaybeCast(die, die_axis->var.dtype()));
      }
      subst.Set(sm_axis->var, MaybeCast(local_cluster, sm_axis->var.dtype()));
    }

    body = Substitute(body, subst);

    if (swizzle_panel.has_value()) {
      PrimExpr swizzle_value =
          Call(DataType::Handle(), tirx::builtin::tvm_tuple(),
               {StringImm(swizzle_order),
                IntImm(DataType::Int(32), swizzle_panel.value())});
      body = AttrStmt(Integer(0), "threadblock_swizzle_pattern", swizzle_value,
                      body);
    }

    if (cluster_dims_.defined()) {
      body = AnnotateLaunchRootBlock(std::move(body), cluster_dims_.value());
    }

    // Emit threadIdx.x and, like T.Kernel, dummy threadIdx.y/threadIdx.z
    // (extent 1). Downstream passes (e.g. ThreadSync) assume all three thread
    // axes are present.
    Var tx_var;
    PrimExpr tx_extent;
    if (thread_axis.has_value()) {
      tx_var = thread_axis->var;
      tx_extent = thread_axis->extent;
    }
    if (tx_var.defined()) {
      IterVar tz = MakeThreadIterVar(Var("tz", DataType::Int(32)), Integer(1),
                                     "threadIdx.z");
      body = AttrStmt(tz, tirx::attr::thread_extent, Integer(1), body);
      IterVar ty = MakeThreadIterVar(Var("ty", DataType::Int(32)), Integer(1),
                                     "threadIdx.y");
      body = AttrStmt(ty, tirx::attr::thread_extent, Integer(1), body);
      IterVar tx = MakeThreadIterVar(tx_var, tx_extent, "threadIdx.x");
      body = AttrStmt(tx, tirx::attr::thread_extent, tx_extent, body);
    }

    if (!block_ivars.empty()) {
      for (auto it = block_ivars.rbegin(); it != block_ivars.rend(); ++it) {
        body = AttrStmt(*it, tirx::attr::thread_extent, (*it)->dom->extent, body);
      }
    } else {
      IterVar block = MakeThreadIterVar(bx, grid_extent, "blockIdx.x");
      body = AttrStmt(block, tirx::attr::thread_extent, grid_extent, body);
    }
    return body;
  }

  Stmt AnnotateLaunchRootBlock(Stmt body, const Array<Integer> &cluster_dims) {
    class RootBlockAnnotator : public StmtMutator {
    public:
      explicit RootBlockAnnotator(Array<Integer> cluster_dims)
          : cluster_dims_(std::move(cluster_dims)) {}

      bool annotated() const { return annotated_; }

    private:
      Stmt VisitStmt_(const SBlockRealizeNode *op) final {
        if (annotated_) {
          return StmtMutator::VisitStmt_(op);
        }
        if (op->block->name_hint == "root") {
          return StmtMutator::VisitStmt_(op);
        }
        SBlock block = op->block;
        auto block_ptr = block.CopyOnWrite();
        block_ptr->annotations.Set("cluster_dims", cluster_dims_);

        auto realize = ffi::GetRef<SBlockRealize>(op);
        auto realize_ptr = realize.CopyOnWrite();
        realize_ptr->block = block;
        annotated_ = true;
        return realize;
      }

      Array<Integer> cluster_dims_;
      bool annotated_{false};
    };

    RootBlockAnnotator annotator(cluster_dims);
    body = annotator(std::move(body));
    ICHECK(annotator.annotated())
        << "T.Scale cluster launch requires a block scope for cluster_dims";
    return body;
  }

  PrimExpr MakeLogicalSm(const ScaleAxis &axis) {
    PrimExpr smid = MakeGetSmid();
    auto schedule_obj = axis.annotations.Get(kScaleSmSchedule);
    if (!schedule_obj) {
      return smid;
    }
    if (auto buffer = schedule_obj.value().try_cast<Buffer>()) {
      Array<PrimExpr> indices;
      indices.push_back(smid);
      return BufferLoad(buffer.value(), indices);
    }
    auto schedule = schedule_obj.value().try_cast<PrimExpr>();
    ICHECK(schedule.has_value())
        << "T.Scale sm_schedule must be a Buffer, BufferLoad, or buffer data Var";
    if (const auto *var = schedule.value().as<VarNode>()) {
      Var data_var = ffi::GetRef<Var>(var);
      auto it = buffer_data_to_buffer_.find(data_var);
      ICHECK(it != buffer_data_to_buffer_.end())
          << "T.Scale sm_schedule data var `" << data_var
          << "` is not present in PrimFunc.buffer_map";
      Array<PrimExpr> indices;
      indices.push_back(smid);
      return BufferLoad((*it).second, indices);
    }
    const auto *load = schedule.value().as<BufferLoadNode>();
    ICHECK(load != nullptr)
        << "T.Scale sm_schedule must be a buffer load, e.g. sm_schedule[T.get_smid()]";
    Array<PrimExpr> indices;
    indices.push_back(smid);
    return BufferLoad(load->buffer, indices, load->predicate, load->span);
  }

  void SetClusterDims(const ScaleAxis &axis) {
    const auto *extent = axis.extent.as<IntImmNode>();
    ICHECK(extent != nullptr)
        << "T.Scale cluster rank extent must be a constant integer";
    ICHECK(extent->value > 0);
    SetClusterDims(Array<Integer>{Integer(extent->value), Integer(1), Integer(1)});
  }

  void SetClusterDims(const Array<Integer> &dims) {
    ICHECK_EQ(dims.size(), 3U);
    if (cluster_dims_.defined()) {
      auto existing = cluster_dims_.value();
      ICHECK_EQ(existing.size(), 3U);
      for (int i = 0; i < 3; ++i) {
        ICHECK_EQ(existing[i]->value, dims[i]->value)
            << "Conflicting cluster dims in T.Scale";
      }
    }
    cluster_dims_ = dims;
  }

  const LaunchGroup *group_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Optional<Array<Integer>> cluster_dims_{std::nullopt};
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass LowerScaleLaunch() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    return ScaleLaunchLowerer::Rewrite(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerScaleLaunch", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.LowerScaleLaunch", LowerScaleLaunch);
}

} // namespace tl
} // namespace tvm
