from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cmax_fused",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cmax_fused.cmax_fused_cuda_ext",
            sources=[
                "cmax_fused/csrc/cmax_fused.cpp",
                "cmax_fused/csrc/cmax_fused_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)