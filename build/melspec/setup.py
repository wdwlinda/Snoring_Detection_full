# from distutils.core import setup, Extension
# from Cython.Build import cythonize

# setup(ext_modules = cythonize(
#         "melspec.cc",                 # our Cython source
#         # sources=["Melspec.cc"],  # additional source file(s)
#         # language="c++",             # generate C++ code
#     ))

# setup(name='melspec',
#       ext_modules=[
#         Extension(
#             'melspec',
#             ['melspec.cc'],
#             include_dirs=['..']
#         )
#       ]
# )

# from setuptools import setup
# from torch.utils.cpp_extension import CppExtension, BuildExtension

# setup(name='melspec',
#       ext_modules=[CppExtension('melspec', ['melspec.cc'])],
#       cmdclass={'build_ext': BuildExtension})


from setuptools import setup

from setuptools_cpp import CMakeExtension, ExtensionBuilder, Pybind11Extension

ext_modules = [
    # A basic pybind11 extension in <project_root>/src/ext1:
    Pybind11Extension(
        "melspec.ext1", ["melspec.cc"], include_dirs=[".."]
    ),
]

setup(
    name="melspec",
    version="0.1.0",
    packages=["melspec"],
    # ... other setup kwargs ...
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=ExtensionBuilder),
    zip_safe=False,
)