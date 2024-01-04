# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Python setup script."""

import argparse
import os
import shutil
import subprocess
import sys

import setuptools
import setuptools.command.build_py
import setuptools.command.install
import wheel.bdist_wheel


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None)
    args, unknown = parser.parse_known_args()
    args.git_version = None
    args.long_description = ""
    sys.argv = [sys.argv[0]] + unknown
    if args.version is None and os.path.exists("version.txt"):
        with open("version.txt", "r") as f:
            args.version = f.read().strip()
    if os.path.exists(".git"):
        try:
            git_version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="./")
            args.git_version = git_version.decode("ascii").strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if os.path.exists("README.md"):
        with open(os.path.join("README.md"), encoding="utf-8") as f:
            args.long_description = f.read()
    return args


def clean_builds():
    """Clean the builds."""
    if os.path.exists("build/lib"):
        shutil.rmtree("build/lib")
    if os.path.exists("flash_attn_dragon.egg-info"):
        shutil.rmtree("flash_attn_dragon.egg-info")


def find_libraries(build_lib):
    """Return the pre-built libraries."""
    in_prefix = "" if sys.platform == "win32" else "lib"
    in_suffix = ".so"
    if sys.platform == "win32":
        in_suffix = ".dll"
    elif sys.platform == "darwin":
        in_suffix = ".dylib"
    libraries = {
        "targets/native/lib/{}dragon_flashattn{}".format(in_prefix, in_suffix): build_lib
        + "/flash_attn_dragon/lib/{}dragon_flashattn{}".format(in_prefix, in_suffix),
    }
    if sys.platform == "win32":
        libraries["targets/native/lib/dragon_flashattn.lib"] = (
            build_lib + "/flash_attn_dragon/lib/dragon_flashattn.lib"
        )
    return libraries


class BuildPyCommand(setuptools.command.build_py.build_py):
    """Enhanced 'build_py' command."""

    def build_packages(self):
        shutil.copytree("flash_attn", self.build_lib + "/flash_attn_dragon")
        with open(self.build_lib + "/flash_attn_dragon/version.py", "w") as f:
            f.write(
                'version = "{}"\n'
                'git_version = "{}"\n'
                "__version__ = version\n".format(args.version, args.git_version)
            )

    def build_package_data(self):
        if not os.path.exists(self.build_lib + "/flash_attn_dragon/lib"):
            os.makedirs(self.build_lib + "/flash_attn_dragon/lib")
        for src, dest in find_libraries(self.build_lib).items():
            if os.path.exists(src):
                shutil.copy(src, dest)
            else:
                print(
                    "ERROR: Unable to find the library at <%s>.\n"
                    "Build it before installing to package." % src
                )
                sys.exit()


class InstallCommand(setuptools.command.install.install):
    """Enhanced 'install' command."""

    def initialize_options(self):
        super(InstallCommand, self).initialize_options()
        self.old_and_unmanageable = True


class WheelDistCommand(wheel.bdist_wheel.bdist_wheel):
    """Enhanced 'bdist_wheel' command."""

    def finalize_options(self):
        super(WheelDistCommand, self).finalize_options()
        self.root_is_pure = False


args = parse_args()
setuptools.setup(
    name="flash-attn-dragon",
    version=args.version,
    description="Flash Attention for Dragon",
    long_description=args.long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seetaresearch/flash-attention",
    author="SeetaTech",
    license="BSD 2-Clause",
    packages=["flash_attn"],
    cmdclass={
        "build_py": BuildPyCommand,
        "install": InstallCommand,
        "bdist_wheel": WheelDistCommand,
    },
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
clean_builds()
