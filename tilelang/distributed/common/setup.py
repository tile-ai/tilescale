from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='ipc_ext',
    ext_modules=[
        CppExtension(
            'ipc_ext', ['ipc_ext.cpp'],
            include_dirs=['/usr/local/cuda/include'],
            extra_compile_args=['-O3'])
    ],
    cmdclass={'build_ext': BuildExtension})
