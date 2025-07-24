# Hierarchical Distributed Architecture (HDA)

``` yaml
# h100.yaml
# Example 1:
# HDA description of DGX-H100 (8x SXM H100s, each with 144 SMs)
# Hardware specification
#   PE: streaming multiprocessor -- with 4 processors, each with 32 cores
#   PE groups:
#     arch_level_0: TPC (Texture Processing Cluster) contains 2 PEs
#     arch_level_1: GPC (Graphics Processing Cluster) contains 9 TPCs
#     arch_level_2: GPU contains 8 GPCs
#     arch_level_3: Node contains 8 GPUs
hw_spec:
  pe:
      - {name: pe_level_0, num_sub_units: 1}
      - {name: pe_level_1, num_sub_units: 32}
      - {name: pe_level_2, num_sub_units: 4}
  pe_groups:
      - {name: arch_level_0, num_sub_units: 2, topology: switch}
      - {name: arch_level_1, num_sub_units: 9, topology: switch}
      - {name: arch_level_2, num_sub_units: 8, topology: switch}
      - {name: arch_level_3, num_sub_units: 8, topology: switch}

hw_abstr:
  - {name: thread, target: pe_level_0}
  - {name: warp, target: pe_level_1}
  - {name: warp_group, target: pe_level_2}
  - {name: block, target: pe_level_2}
  - {name: cta_pair, target: arch_level_0}
  - {name: tbc, target: arch_level_1}
  - {name: gpu, target: arch_level_2}
  - {name: node, target: arch_level_3}

hints:
  warp:
    - lambda _scales: _scales.thread // 32
  warp_group:
    - lambda _scales: _scales.warp // 4
    - lambda _scales: _scales.thread // 128
  block:
    - lambda _scales: _scales.cta_pair * 2
  cta_pair:
    - ...

constrains:
  - lambda _scales: _scales.warp_group == _scales.warp // 4
  - lambda _scales: _scales.warp == _scales.thread // 32
  - ...
```
``` python
from dataclasses import dataclass

@dataclass
class ComputeUnit():
    name: str
    num_sub_units: int
    topology: str  # switch, link, ring

@dataclass
class ScaleView():
    name: str
    target: str
    hints_lambdas: list[str]
    constrains_lambdas: list[str]
    instance_num: int
    topology: str
    is_flatten: bool = False

@dataclass
class HardwareSpecification:
    pe: list[ComputeUnit]
    pe_groups: list[ComputeUnit]

@dataclass
class HardwareAbstraction:
    scales: list[ScaleView]

@dataclass
class ScalesInstance(HardwareAbstraction):
    def remove_scale(self, scale_name: str) -> tuple['ScalesInstance', ScaleView]:
        ...

    def insert_scale(self, scale_name: str, scale_layer: int, topology: str) -> 'ScalesInstance':
        ...

@dataclass
class HDA:
    hw_spec: HardwareSpecification
    hw_abstr: HardwareAbstraction

    @staticmethod
    def from_yaml(yaml_file: str) -> 'HDA':
        ...

    def instantiate(self, scale_views: list[ScaleView]) -> ScalesInstance:
        ...
```

``` python
h100_hda = HDA.from_yaml("h100.yaml")
# Example 1: Launch 512 blocks per GPU, each block contains 256 threads
launched_instance = h100_hda.instantiate(scale_views=[
    ScaleView(name="gpu", instance_num=-1),                         # -1 means auto-inferring from the hw_spec
    ScaleView(name="block", instance_num=512, is_flatten=True),     # flatten means collapsing the intermediate scales
    ScaleView(name="thread", instance_num=256, is_flatten=True),
])
with launched_instance:
    with T.Scale("block"):
        # Although the warp_group is not explicitly specified, it is automatically inferred from the hints
        with T.Scale("warp_group"):
            with T.Scale("thread"):
                ...
# Example 2: Launch 4096 blocks for the entire node, each block contains 256 threads
launched_instance = h100_hda.instantiate(scale_views=[
    ScaleView(name="block", instance_num=4096, is_flatten=True),
    ScaleView(name="thread", instance_num=256, is_flatten=True),
])
with launched_instance:
    with T.Scale("block"):
        with T.Scale("thread"):
            ...
# Example 3: Launch only a single block for each SM, each block contains 256 threads
launched_instance = h100_hda.instantiate(scale_views=[
    ScaleView(name="block", instance_num=-1, is_flatten=True),
    ScaleView(name="thread", instance_num=256, is_flatten=True),
])
with launched_instance:
    with T.Scale("block"):
        with T.Scale("thread"):
            ...
# Example 4: Launch only a single block for each SM, each block contains 3 warp groups
launched_instance = h100_hda.instantiate(scale_views=[
    ScaleView(name="block", instance_num=-1, is_flatten=True),
    ScaleView(name="warp_group", instance_num=3),
    ScaleView(name="warp", instance_num=4),
    ScaleView(name="thread", instance_num=32),
])
with launched_instance:
    with T.Scale("block"):
        with T.Scale("warp_group"):
            with T.Scale("warp"):
                with T.Scale("thread"):
                    ...
# Example 5: Launch a kernel with a virtualized 2D mesh to perform a connon-style matrix multiplication
launched_instance = h100_hda.instantiate(scale_views=[
    ScaleView(name="block", instance_num=-1, is_flatten=True),
    ScaleView(name="warp_group", instance_num=3),
    ScaleView(name="warp", instance_num=4),
    ScaleView(name="thread", instance_num=32),
])
tmp_instance, node_layer = launched_instance.remove_scale("node")
node_x_layer, node_y_layer = node_layer.split([4, 4], ["link", "link"])
launched_instance = tmp_instance.insert_scale("node_x_layer", -1).insert_scale("node_y_layer", -1)

with launched_instance:
    with T.Scale("node_x_layer"):
        with T.Scale("node_y_layer"):
            ...  # 2D mesh of virtualized 4x4 nodes
```
