#!/usr/bin/env bash
# Legacy host-install path. The supported way to set up an ABIDES environment
# for the course is via Docker — see the "Quickstart for Students" section of
# the README. This script is kept for users who specifically need a host
# install.
#
# Requirements:
#   - Python 3.11 (matches the Docker image; 3.9/3.10 should also work).
#   - C/C++ build tools for numba, llvmlite, pomegranate
#       macOS: xcode-select --install
#       Ubuntu/Debian: sudo apt install build-essential
#       Windows: install Visual Studio "Build Tools for C++" (or just use Docker)
set -euo pipefail

python3 -m pip install --upgrade pip setuptools wheel

# Prefer the pinned lockfile when present; otherwise fall back to the
# direct-deps files (slower resolution, but works on a fresh clone).
if [ -f requirements.lock ]; then
    python3 -m pip install -r requirements.lock
else
    python3 -m pip install -r requirements.in -r requirements-dev.in
fi

# Editable installs of the three sub-packages. PEP 660 editable installs
# (pip 21.3+) read from each sub-package's pyproject.toml.
python3 -m pip install -e ./abides-core
python3 -m pip install -e ./abides-markets
python3 -m pip install -e ./abides-gym
