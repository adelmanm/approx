from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='approx_conv2d_cpp',
      ext_modules=[CppExtension('approx_conv2d', ['approx_conv2d.cpp'], extra_link_args=['-L/usr/lib/x86_64-linux-gnu/'])],
      cmdclass={'build_ext': BuildExtension})
setup(name='approx_linear_xA_b_cpp',
      ext_modules=[CppExtension('approx_linear_xA_b', ['approx_linear_xA_b.cpp'], extra_link_args=['-L/usr/lib/x86_64-linux-gnu/'])],
      cmdclass={'build_ext': BuildExtension})
