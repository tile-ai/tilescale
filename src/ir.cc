/*!
 * \file tl/ir.cc
 * \brief Extension for the tvm script frontend.
 *
 */

#include "./transform/common/attr.h"
#include "op/builtin.h"
#include "tvm/ffi/any.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/script/ir_builder/tir/ir.h>

#include <utility>

namespace tvm {
namespace tl {

using namespace script::ir_builder::tir;

static Var CreateEnvThread(String name, String thread_tag, DataType dtype) {
  using namespace tvm::tir;
  using namespace tvm::script::ir_builder;
  IterVar iter_var(Range{nullptr}, Var(std::move(name), dtype),
                   tvm::tir::IterVarType::kThreadIndex, std::move(thread_tag));
  Var var = iter_var->var;
  if (Optional<PrimFuncFrame> opt_frame =
          IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    opt_frame.value()->env_threads.Set(var, iter_var);
  } else {
    LOG(FATAL) << "EnvThread can only be used inside a PrimFunc";
  }
  return var;
}

static ForFrame MakeIterVarFrame(const std::string &name, const PrimExpr &dom) {
  using namespace tvm::tir;
  Var var = Var(name, dom->dtype);
  // Create a frame that represents a loop over the given domain.
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.push_back(var);
  n->doms.push_back(Range(0, dom));
  n->f_make_for_loop = [](const Array<Var> &vars, const Array<Range> &doms,
                          const Stmt &body) -> Stmt {
    ICHECK_EQ(vars.size(), 1);
    ICHECK_EQ(doms.size(), 1);
    return For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kSerial, body);
  };
  return ForFrame(n);
}

ForFrame ParallelFor(const Array<PrimExpr> &extents,
                     const Map<String, ObjectRef> &annotations) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  for (const auto &extent : extents) {
    DataType dtype = extent.dtype();
    n->vars.push_back(Var("v", extent.dtype()));
    n->doms.push_back(Range(make_const(dtype, 0), extent));
  }
  n->f_make_for_loop = [annotations](const Array<Var> &vars,
                                     const Array<Range> &doms,
                                     Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      body = For(var, dom->min, dom->extent, ForKind::kParallel, body,
                 /*thread_binding=*/std::nullopt, /*annotations=*/annotations);
    }
    return body;
  };
  return ForFrame(n);
}

ForFrame PipelinedFor(PrimExpr start, const PrimExpr &stop, int num_stages,
                      const Array<PrimExpr> &order,
                      const Array<PrimExpr> &stages,
                      const Array<Array<PrimExpr>> &sync,
                      const Array<Array<PrimExpr>> &groups) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  DataType dtype = stop.dtype();
  n->vars.push_back(Var("v", dtype));
  n->doms.push_back(Range(std::move(start), stop));
  n->f_make_for_loop = [=](const Array<Var> &vars, const Array<Range> &doms,
                           Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    ICHECK(n == 1);
    Map<String, ObjectRef> anno;
    if (num_stages > 0)
      anno.Set("num_stages", PrimExpr(num_stages));
    if (!order.empty())
      anno.Set("tl_pipeline_order", order);
    if (!stages.empty())
      anno.Set("tl_pipeline_stage", stages);
    if (!sync.empty())
      anno.Set("tl_pipeline_sync", sync);
    if (!groups.empty())
      anno.Set("tl_pipeline_group", groups);
    body = For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kSerial, body,
               /*thread_binding=*/std::nullopt, /*annotations=*/anno);
    return body;
  };
  return ForFrame(n);
}

ForFrame PersistentFor(const Array<PrimExpr> &domain, const PrimExpr &wave_size,
                       const PrimExpr &index, PrimExpr group_size) {
  using namespace tvm::tir;
  ICHECK(!domain.empty());
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(domain.size());
  n->doms.reserve(domain.size());
  PrimExpr domain_size = domain[0];
  for (int i = 1; i < domain.size(); i++) {
    domain_size *= domain[i];
  }

  auto waves = ceildiv(domain_size, wave_size);
  auto loop_var = Var("w", waves.dtype());
  group_size = min(group_size, domain[domain.size() - 1]);
  Array<Var> coord_vars;

  for (int i = 0; i < domain.size(); ++i) {
    DataType dtype = domain[i].dtype();
    Var coord("v" + std::to_string(i), dtype);
    coord_vars.push_back(coord);
    n->vars.push_back(coord);
    n->doms.push_back(Range(make_const(dtype, 0), domain[i]));
  }

  Array<PrimExpr> grouped_domain;
  grouped_domain.push_back(truncdiv(domain[domain.size() - 1], group_size));
  for (int i = 0; i < domain.size() - 1; ++i) {
    grouped_domain.push_back(domain[i]);
  }
  grouped_domain.push_back(group_size);

  n->f_make_for_loop = [=](const Array<Var> &vars, const Array<Range> &doms,
                           const Stmt &body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    Map<String, ObjectRef> anno;
    Array<PrimExpr> idxs(grouped_domain.size(), PrimExpr());
    PrimExpr rem = loop_var * wave_size + index;

    for (int i = grouped_domain.size() - 1; i >= 1; --i) {
      idxs.Set(i, truncmod(rem, grouped_domain[i]));
      rem = truncdiv(rem, grouped_domain[i]);
    }
    idxs.Set(0, rem);

    auto out_if = tvm::tir::IfThenElse(
        domain_size <= (loop_var * wave_size + index),
        tvm::tir::Evaluate(
            tvm::tir::Call(DataType::Handle(), tvm::tl::loop_break(), {})),
        Stmt());

    arith::Analyzer analyzer;
    Stmt new_body = body;
    if (analyzer.CanProveGreaterEqual(waves, 2)) {
      new_body = SeqStmt({out_if, body});
    }
    Stmt outer =
        For(loop_var, 0, waves, ForKind::kSerial, new_body, std::nullopt, anno);
    for (int i = 0; i < vars.size() - 1; ++i) {
      outer = tvm::tir::LetStmt(vars[i], idxs[i + 1], outer);
    }
    outer = tvm::tir::LetStmt(vars[vars.size() - 1],
                              idxs[0] * group_size + idxs[vars.size()], outer);
    return outer;
  };

  return ForFrame(n);
}

/*!
 * \brief A frame that represents a kernel launch.
 *
 * \sa KernelLaunchFrameNode
 */
class KernelLaunchFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<KernelLaunchFrameNode>().def_ro(
        "frames", &KernelLaunchFrameNode::frames);
  }

  static constexpr const char *_type_key = "tl.KernelLaunchFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(KernelLaunchFrameNode, TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    for (auto frame = frames.begin(); frame != frames.end(); ++frame)
      (*frame)->EnterWithScope();
  }
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  TVM_DLL void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
      (*frame)->ExitWithScope();
  }
};

/*!
 * \brief Managed reference to KernelLaunchFrameNode.
 *
 * \sa KernelLaunchFrameNode
 */
class KernelLaunchFrame : public TIRFrame {
public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(KernelLaunchFrame, TIRFrame,
                                                    KernelLaunchFrameNode);
};

KernelLaunchFrame KernelLaunch(const Array<PrimExpr> &grid_size,
                               const Optional<Array<PrimExpr>> &block_size_opt,
                               const Map<String, ffi::Any> &attrs) {
  ObjectPtr<KernelLaunchFrameNode> n = make_object<KernelLaunchFrameNode>();

  // If the kernel is a CPU kernel, we don't need to launch any threads.
  bool is_cpu_kernel_frame =
      attrs.defined() && attrs.count(tilelang_is_cpu_kernel_frame);

  auto block_size = block_size_opt.value_or(Array<PrimExpr>());

  if (is_cpu_kernel_frame) {
    // Launch CPU Kernel
    ICHECK(grid_size.size() >= 0);
    ICHECK(block_size.empty()) << "CPU kernel cannot have block size";
    ICHECK(attrs.defined());
    // create grid loop var
    for (int i = 0; i < grid_size.size(); i++) {
      n->frames.push_back(
          MakeIterVarFrame("block_var_" + std::to_string(i), grid_size[i]));
    }
  } else {
    // Launch GPU Kernel
    ICHECK(grid_size.size() <= 3);
    if (!grid_size.empty())
      n->frames.push_back(LaunchThread(
          CreateEnvThread("bx", "blockIdx.x", grid_size[0].dtype()),
          grid_size[0]));
    if (grid_size.size() > 1)
      n->frames.push_back(LaunchThread(
          CreateEnvThread("by", "blockIdx.y", grid_size[1].dtype()),
          grid_size[1]));
    if (grid_size.size() > 2)
      n->frames.push_back(LaunchThread(
          CreateEnvThread("bz", "blockIdx.z", grid_size[2].dtype()),
          grid_size[2]));
    if (block_size.defined()) {
      ICHECK(block_size.size() <= 3);
      if (!block_size.empty()) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("tx", "threadIdx.x", block_size[0].dtype()),
            block_size[0]));
      }
      if (block_size.size() > 1) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("ty", "threadIdx.y", block_size[1].dtype()),
            block_size[1]));
      }
      if (block_size.size() > 2) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("tz", "threadIdx.z", block_size[2].dtype()),
            block_size[2]));
      }
    }
  }

  if (attrs.defined()) {
    auto empty_block = tvm::script::ir_builder::tir::Block(MainBlockName);
    empty_block->annotations = attrs;
    n->frames.push_back(empty_block);
  } else {
    n->frames.push_back(tvm::script::ir_builder::tir::Block(MainBlockName));
  }

  return KernelLaunchFrame(n);
}

// A scope kernel launch that separates logical grid and cluster shapes, but
// lowers to a standard KernelLaunch with blocks = grid[i] * cluster[i].
KernelLaunchFrame ScopeKernelLaunch(const Array<PrimExpr> &grid_size,
                                    const Optional<Array<PrimExpr>> &cluster_size_opt,
                                    const Optional<Array<PrimExpr>> &block_size_opt,
                                    const Map<String, ffi::Any> &attrs) {
  ObjectPtr<KernelLaunchFrameNode> n = make_object<KernelLaunchFrameNode>();

  bool is_cpu_kernel_frame =
      attrs.defined() && attrs.count(tilelang_is_cpu_kernel_frame);

  // Normalize cluster size to up to 3 dims, defaulting to 1s
  Array<PrimExpr> cluster_size = cluster_size_opt.value_or(Array<PrimExpr>());

  auto get_cluster = [&](int i, DataType dt) -> PrimExpr {
    if (cluster_size.defined() && i < static_cast<int>(cluster_size.size()))
      return cluster_size[i];
    return make_const(dt, 1);
  };

  if (is_cpu_kernel_frame) {
    // CPU: Only serial grid loops
    for (int i = 0; i < static_cast<int>(grid_size.size()); i++) {
      n->frames.push_back(
          MakeIterVarFrame("block_var_" + std::to_string(i), grid_size[i]));
    }
  } else {
    if (cluster_size.defined()) {
      ICHECK(cluster_size.size() <= 3);
      if (!cluster_size.empty()) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("cx", "clusterIdx.x", cluster_size[0].dtype()),
            cluster_size[0]));
      }
      if (cluster_size.size() > 1) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("cy", "clusterIdx.y", cluster_size[1].dtype()),
            cluster_size[1]));
      }
      if (cluster_size.size() > 2) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("cz", "clusterIdx.z", cluster_size[2].dtype()),
            cluster_size[2]));
      }
    }

    ICHECK(grid_size.size() <= 3);
    if (!grid_size.empty()) {
      PrimExpr extent = grid_size[0];
      n->frames.push_back(LaunchThread(
          CreateEnvThread("bx", "blockIdx.x", extent.dtype()), extent));
    }
    if (grid_size.size() > 1) {
      PrimExpr extent = grid_size[1];
      n->frames.push_back(LaunchThread(
          CreateEnvThread("by", "blockIdx.y", extent.dtype()), extent));
    }
    if (grid_size.size() > 2) {
      PrimExpr extent = grid_size[2];
      n->frames.push_back(LaunchThread(
          CreateEnvThread("bz", "blockIdx.z", extent.dtype()), extent));
    }

    if (block_size_opt.defined()) {
      const auto &block_size = block_size_opt.value();
      ICHECK(block_size.size() <= 3);
      if (!block_size.empty()) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("tx", "threadIdx.x", block_size[0].dtype()),
            block_size[0]));
      }
      if (block_size.size() > 1) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("ty", "threadIdx.y", block_size[1].dtype()),
            block_size[1]));
      }
      if (block_size.size() > 2) {
        n->frames.push_back(LaunchThread(
            CreateEnvThread("tz", "threadIdx.z", block_size[2].dtype()),
            block_size[2]));
      }
    }
  }

  // Attach a terminal block to carry annotations including cluster/grid shapes
  auto empty_block = tvm::script::ir_builder::tir::Block(MainBlockName);
  Map<String, ffi::Any> annot;
  if (attrs.defined()) {
    annot = attrs;
  }

  // Normalize and attach grid/cluster shapes as annotations for Python helpers
  Array<PrimExpr> grid_shape;
  Array<PrimExpr> cluster_shape;
  grid_shape.reserve(3);
  cluster_shape.reserve(3);
  for (int i = 0; i < 3; ++i) {
    PrimExpr g = (i < static_cast<int>(grid_size.size()))
                     ? grid_size[i]
                     : make_const(DataType::Int(32), 1);
    DataType dt = g.dtype();
    grid_shape.push_back(g);
    cluster_shape.push_back(get_cluster(i, dt));
  }
  annot.Set("tilelang.grid_shape", grid_shape);
  annot.Set("tilelang.cluster_shape", cluster_shape);

  empty_block->annotations = annot;

  n->frames.push_back(empty_block);

  return KernelLaunchFrame(n);
}

TVM_REGISTER_NODE_TYPE(KernelLaunchFrameNode);

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.Parallel", ParallelFor)
      .def("tl.Pipelined", PipelinedFor)
      .def("tl.Persistent", PersistentFor)
      .def("tl.KernelLaunch", KernelLaunch)
      .def("tl.ScopeKernelLaunch", ScopeKernelLaunch);
});

class WarpSpecializeFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WarpSpecializeFrameNode>().def_ro(
        "frames", &WarpSpecializeFrameNode::frames);
  }

  static constexpr const char *_type_key = "tl.WarpSpecializeFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(WarpSpecializeFrameNode, TIRFrameNode);

public:
  TVM_DLL void EnterWithScope() final {
    for (auto frame = frames.begin(); frame != frames.end(); ++frame)
      (*frame)->EnterWithScope();
  }
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  TVM_DLL void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame)
      (*frame)->ExitWithScope();
  }
};

class WarpSpecializeFrame : public TIRFrame {
public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(WarpSpecializeFrame,
                                                    TIRFrame,
                                                    WarpSpecializeFrameNode);
};

WarpSpecializeFrame WarpSpecialize(const Array<IntImm> &warp_group_ids,
                                   const PrimExpr &thread_idx,
                                   int warp_group_size = 128) {
  ObjectPtr<WarpSpecializeFrameNode> n = make_object<WarpSpecializeFrameNode>();
  PrimExpr condition;
  std::vector<int> warp_groups;
  warp_groups.reserve(warp_group_ids.size());
  for (int i = 0; i < warp_group_ids.size(); i++) {
    warp_groups.push_back(Downcast<IntImm>(warp_group_ids[i])->value);
  }
  std::sort(warp_groups.begin(), warp_groups.end());

  // Merge consecutive groups
  std::vector<std::pair<int, int>> merged;
  for (int group : warp_groups) {
    if (merged.empty() || group != merged.back().second) {
      merged.emplace_back(group, group + 1);
    } else {
      merged.back().second = group + 1;
    }
  }

  for (const auto &[start, end] : merged) {
    PrimExpr min_bound = IntImm(thread_idx.dtype(), start) * warp_group_size;
    PrimExpr max_bound = IntImm(thread_idx.dtype(), end) * warp_group_size;
    PrimExpr range_cond = (thread_idx >= min_bound) && (thread_idx < max_bound);

    if (condition.defined()) {
      condition = tir::Or(condition, range_cond);
    } else {
      condition = range_cond;
    }
  }
  IfFrame if_frame = If(condition);
  AttrFrame attr_frame = Attr(Integer(0), "warp_specialize", Integer(1));
  n->frames.push_back(if_frame);
  n->frames.push_back(Then());
  n->frames.push_back(attr_frame);
  return WarpSpecializeFrame(n);
}

TVM_REGISTER_NODE_TYPE(WarpSpecializeFrameNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.WarpSpecialize", WarpSpecialize);
  KernelLaunchFrameNode::RegisterReflection();
  WarpSpecializeFrameNode::RegisterReflection();
});

} // namespace tl
} // namespace tvm
