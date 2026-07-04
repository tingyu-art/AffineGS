import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6")

setup(
    name='gaussian_flow_renderer',
    ext_modules=[
        CUDAExtension(
            name='gaussian_flow_renderer_ext',
            sources=['renderer.cpp', 'renderer_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                # no --use_fast_math: keep float32 math behavior strict
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
