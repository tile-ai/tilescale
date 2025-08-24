# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='set_value',
    ext_modules=[CUDAExtension(
        name='set_value',
        sources=['set_value.cu'],
    )],
    cmdclass={'build_ext': BuildExtension})
