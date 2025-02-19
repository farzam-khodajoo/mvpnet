"""Setup extension

Notes:
    If extra_compile_args is provided, you need to provide different instances for different extensions.
    Refer to https://github.com/pytorch/pytorch/issues/20169

"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='ext',
    ext_modules=[
        CUDAExtension(
            name='fps_cuda',
            sources=[
                'cuda/fps.cpp',
                'cuda/fps_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g', '/Ox'], 'nvcc': ['-O0']}
        ),
        CUDAExtension(
            name='group_points_cuda',
            sources=[
                'cuda/group_points.cpp',
                'cuda/group_points_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g', '/Ox'], 'nvcc': ['-O0']}
        ),
        CUDAExtension(
            name='ball_query_cuda',
            sources=[
                'cuda/ball_query.cpp',
                'cuda/ball_query_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g', '/Ox'], 'nvcc': ['-O0']}
        ),
        CUDAExtension(
            name='ball_query_distance_cuda',
            sources=[
                'cuda/ball_query_distance.cpp',
                'cuda/ball_query_distance_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g', '/Ox'], 'nvcc': ['-O0']}
        ),
        CUDAExtension(
            name='knn_distance_cuda',
            sources=[
                'cuda/knn_distance.cpp',
                'cuda/knn_distance_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g', '/Ox'], 'nvcc': ['-O0']}
        ),
        CUDAExtension(
            name='interpolate_cuda',
            sources=[
                'cuda/interpolate.cpp',
                'cuda/interpolate_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g', '/Ox'], 'nvcc': ['-O0']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
