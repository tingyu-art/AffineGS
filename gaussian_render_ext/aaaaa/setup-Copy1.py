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
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
