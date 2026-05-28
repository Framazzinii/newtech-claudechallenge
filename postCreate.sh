#!/usr/bin/env bash
# Runs once after the dev container is built.
# Idempotent: safe to re-run.
set -euo pipefail

cd /abides

echo "[postCreate] Installing ABIDES sub-packages in editable mode..."
pip install --no-cache-dir -e ./abides-core    >/dev/null
pip install --no-cache-dir -e ./abides-markets >/dev/null
pip install --no-cache-dir -e ./abides-gym     >/dev/null

echo "[postCreate] Sanity-checking imports..."
python - <<'PY'
import importlib
for mod in ("abides_core", "abides_markets", "abides_gym"):
    importlib.import_module(mod)
    print(f"  ok: {mod}")
PY

echo
echo "Environment ready."
echo "  - Smoke test:   make smoke"
echo "  - Unit tests:   make test"
echo "  - JupyterLab:   make up      (then open http://localhost:8888)"
