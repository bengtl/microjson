#!/usr/bin/env python3
"""Build Cython extensions for 3D tile generation.

Usage::

    .venv/bin/python scripts/build_cython.py

Requires ``cython`` to be installed (``pip install cython``).
Builds ``.so`` / ``.pyd`` files in-place next to the ``.pyx`` sources.
Does NOT modify the poetry-core build system.
"""

import os
import sys
from pathlib import Path

try:
    from Cython.Build import cythonize
except ImportError:
    print("ERROR: Cython not installed. Run: .venv/bin/pip install 'cython>=3.0'")
    sys.exit(1)

from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext

# Root of the project
ROOT = Path(__file__).resolve().parent.parent
PYX_DIR = ROOT / "src" / "microjson" / "tiling3d"

# Collect .pyx files
pyx_files = sorted(PYX_DIR.glob("*.pyx"))
if not pyx_files:
    print("No .pyx files found in", PYX_DIR)
    sys.exit(1)

print(f"Found {len(pyx_files)} Cython source(s):")
for f in pyx_files:
    print(f"  {f.name}")

# Build Extension objects
extensions = []
for pyx in pyx_files:
    # Module name relative to src/
    mod_name = f"microjson.tiling3d.{pyx.stem}"
    extensions.append(Extension(mod_name, [str(pyx)]))

# Cythonize with compiler directives
extensions = cythonize(
    extensions,
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "language_level": "3",
    },
)

# Build in-place
dist = Distribution({"ext_modules": extensions})
dist.package_dir = {"": str(ROOT / "src")}

cmd = build_ext(dist)
cmd.inplace = True
cmd.ensure_finalized()
cmd.run()

print("\nBuild complete. Verifying imports...")

# Verify
sys.path.insert(0, str(ROOT / "src"))
from microjson.tiling3d._accel import CYTHON_AVAILABLE

if CYTHON_AVAILABLE:
    print("CYTHON_AVAILABLE = True")
else:
    print("WARNING: CYTHON_AVAILABLE = False (some extensions may have failed)")
    sys.exit(1)
