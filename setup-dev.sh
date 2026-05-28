#!/usr/bin/env bash
# Legacy host-install path for ABIDES contributors who want an editable
# install without Docker. See the "Quickstart for Students" section of the
# README for the supported (Docker) flow.
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel

if [ -f requirements.lock ]; then
    python -m pip install -r requirements.lock
else
    python -m pip install -r requirements.in -r requirements-dev.in
fi

python -m pip install -e ./abides-core
python -m pip install -e ./abides-markets
python -m pip install -e ./abides-gym
