from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gaussian_render_ext',
    ext_modules=[
        CUDAExtension(
            name='gaussian_render_ext',
            sources=['gaussian_render.cpp', 'gaussian_render_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                # no --use_fast_math: keep float32 math behavior strict
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
