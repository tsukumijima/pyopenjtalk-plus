import os
import subprocess
import sys
from glob import glob
from itertools import chain
from os.path import exists, join
from subprocess import run

import numpy as np
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.develop
from setuptools import Extension, find_packages, setup

platform_is_windows = sys.platform == "win32"

version = "0.3.4-post3"

msvc_extra_compile_args_config = [
    "/source-charset:utf-8",
    "/execution-charset:utf-8",
]


def msvc_extra_compile_args(compile_args):
    cas = set(compile_args)
    xs = filter(lambda x: x not in cas, msvc_extra_compile_args_config)
    return list(chain(compile_args, xs))


class custom_build_ext(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        compiler_type_is_msvc = self.compiler.compiler_type == "msvc"
        for entry in self.extensions:
            if compiler_type_is_msvc:
                entry.extra_compile_args = msvc_extra_compile_args(
                    entry.extra_compile_args if hasattr(entry, "extra_compile_args") else []
                )

        setuptools.command.build_ext.build_ext.build_extensions(self)


def check_cmake_in_path():
    try:
        result = subprocess.run(
            ["cmake", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            # CMake is in the system path
            return True, result.stdout.strip()
        else:
            # CMake is not in the system path
            return False, None
    except FileNotFoundError:
        # CMake command not found
        return False, None


if os.name == "nt":  # Check if the OS is Windows
    # Check if CMake is in the system path
    cmake_found, cmake_version = check_cmake_in_path()

    if cmake_found:
        print(
            f"CMake is in the system path. Version: \
              {cmake_version}"
        )
    else:
        raise SystemError(
            "CMake is not found in the \
                          system path. Make sure CMake \
                          is installed and in the system \
                          path."
        )

# open_jtalk sources
src_top = join("lib", "open_jtalk", "src")

# generate config.h for mecab
# NOTE: need to run cmake to generate config.h
# we could do it on python side but it would be very tricky,
# so far let's use cmake tool
if not exists(join(src_top, "mecab", "src", "config.h")):
    cwd = os.getcwd()
    build_dir = join(src_top, "build")
    os.makedirs(build_dir, exist_ok=True)
    os.chdir(build_dir)

    # NOTE: The wrapped OpenJTalk does not depend on HTS_Engine,
    # but since HTSEngine is included in CMake's dependencies, it refers to a dummy path.
    r = run(["cmake", "..", "-DHTS_ENGINE_INCLUDE_DIR=.", "-DHTS_ENGINE_LIB=dummy"])
    r.check_returncode()
    os.chdir(cwd)

all_src = []
include_dirs = []
for s in [
    "jpcommon",
    "mecab/src",
    "mecab2njd",
    "njd",
    "njd2jpcommon",
    "njd_set_accent_phrase",
    "njd_set_accent_type",
    "njd_set_digit",
    "njd_set_long_vowel",
    "njd_set_pronunciation",
    "njd_set_unvoiced_vowel",
    "text2mecab",
]:
    all_src += glob(join(src_top, s, "*.c"))
    all_src += glob(join(src_top, s, "*.cpp"))
    include_dirs.append(join(os.getcwd(), src_top, s))

# Extension for OpenJTalk frontend
ext_modules = [
    Extension(
        name="pyopenjtalk.openjtalk",
        sources=[join("pyopenjtalk", "openjtalk.pyx")] + all_src,
        include_dirs=[np.get_include()] + include_dirs,
        extra_compile_args=[],
        extra_link_args=[],
        language="c++",
        define_macros=[
            ("HAVE_CONFIG_H", None),
            ("DIC_VERSION", "102"),
            ("MECAB_DEFAULT_RC", '"dummy"'),
            ("PACKAGE", '"open_jtalk"'),
            ("VERSION", '"1.11"'),
            ("CHARSET_UTF_8", None),
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ],
    )
]

# Extension for HTSEngine backend
htsengine_src_top = join("lib", "hts_engine_API", "src")
all_htsengine_src = glob(join(htsengine_src_top, "lib", "*.c"))
ext_modules += [
    Extension(
        name="pyopenjtalk.htsengine",
        sources=[join("pyopenjtalk", "htsengine.pyx")] + all_htsengine_src,
        include_dirs=[np.get_include(), join(htsengine_src_top, "include")],
        extra_compile_args=[],
        extra_link_args=[],
        libraries=["winmm"] if platform_is_windows else [],
        language="c++",
        define_macros=[
            ("AUDIO_PLAY_NONE", None),
        ],
    )
]

# Adapted from https://github.com/pytorch/pytorch
cwd = os.path.dirname(os.path.abspath(__file__))
if os.getenv("PYOPENJTALK_BUILD_VERSION"):
    version = os.getenv("PYOPENJTALK_BUILD_VERSION")
else:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
        # version += "+" + sha[:7]
    except subprocess.CalledProcessError:
        pass
    except IOError:  # FileNotFoundError for python 3
        pass


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        global version, cwd
        print("-- Building version " + version)
        version_path = os.path.join(cwd, "pyopenjtalk", "version.py")
        with open(version_path, "w") as f:
            f.write(f"__version__ = '{version}'\n")


class develop(setuptools.command.develop.develop):
    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)


with open("README.md", "r", encoding="utf8") as fd:
    long_description = fd.read()

setup(
    name="pyopenjtalk-plus",
    version=version,
    description="A Python wrapper for OpenJTalk with additional improvements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ryuichi Yamamoto",
    author_email="zryuichi@gmail.com",
    url="https://github.com/tsukumijima/pyopenjtalk-plus",
    license="MIT",
    packages=find_packages(),
    package_data={
        "": [
            "dictionary/COPYING",
            "dictionary/*.bin",
            "dictionary/*.dic",
            "dictionary/left-id.def",
            "dictionary/pos-id.def",
            "dictionary/rewrite.def",
            "dictionary/right-id.def",
            "htsvoice/*",
            "yomi_model/*",
        ],
        "pyopenjtalk": ["py.typed", "*.pyi"],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": custom_build_ext, "build_py": build_py, "develop": develop},
    install_requires=[
        "importlib_resources; python_version<'3.9'",
        "numpy>=1.24.0, <2.0",
        "sudachipy",
        "sudachidict_core",
        # "scikit-learn>=0.24.2",
        # "pandas>=2.0.1",
    ],
    tests_require=["nose", "coverage"],
    extras_require={
        "docs": [
            "sphinx_rtd_theme",
            "nbsphinx>=0.8.6",
            "Jinja2>=3.0.1",
            "pandoc",
            "ipython",
            "jupyter",
        ],
        "dev": [
            "cython>=0.28.0, <3.0",
            "pysen",
            "taskipy",
            "types-setuptools",
            "black>=19.19b0",
            "click",
            "flake8>=3.7",
            "flake8-bugbear",
            "isort>=4.3",
            "types-decorator",
            "importlib-metadata<5.0",
        ],
        "test": ["pytest"],
        "marine": ["marine-plus"],
    },
    classifiers=[
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Cython",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    keywords=["OpenJTalk", "Research"],
)
