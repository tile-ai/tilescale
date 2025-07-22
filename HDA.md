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
      - {name: pe_level_0, num_units: 1}
      - {name: pe_level_1, num_units: 32}
      - {name: pe_level_2, num_units: 4}
  pe_groups:
      - {name: arch_level_0, num_units: 2}
      - {name: arch_level_1, num_units: 9}
      - {name: arch_level_2, num_units: 8}
      - {name: arch_level_3, num_units: 8}

hw_abstr:
  - {name: thread, target: pe_level_0}
  - {name: warp, target: pe_level_1, infer_lambda: 'lambda _t, _b: _b // 32'}
  - {name: warp_group, target: pe_level_2, infer_lambda: 'lambda _t, _b: _b // 4'}
  - {name: block, target: pe_level_2, }
  - {name: cta_pair, target: arch_level_0}
  - {name: tbc, target: arch_level_1}
  - {name: gpu, target: arch_level_2}
  - {name: node, target: arch_level_3}
```
``` python
from dataclasses import dataclass

@dataclass
class ComputeUnit():
    name: str
    num_units: int

@dataclass
class ScaleView():
    name: str
    target: str
    infer_lambda: str
    instance_num: int
    is_flatten: bool = False

@dataclass
class HardwareSpecification:
    pe: list[ComputeUnit]
    pe_groups: list[ComputeUnit]

@dataclass
class HardwareAbstraction:
    scales: list[ScaleView]

@dataclass
class HDA:
    hw_spec: HardwareSpecification
    hw_abstr: HardwareAbstraction

    @staticmethod
    def from_yaml(yaml_file: str) -> 'HDA':
        ...

    def instantiate(self, scale_views: list[ScaleView]) -> Instance:
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
```

``` yaml
# Example 2:
# HDA description of Cerebras
```
