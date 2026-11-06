#!/usr/bin/env bash
# One-command setup for the Q3 scaffold.
#
#   ./scripts/setup.sh
#
# Creates .venv with uv and installs the light dependencies.
# Takes about 45 seconds on a normal connection.
set -euo pipefail

cd "$(dirname "$0")/.."

UV="${UV:-uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if ! command -v "$UV" >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:"
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "Or: brew install uv"
  exit 1
fi

echo "==> Creating virtual environment (.venv) with Python ${PYTHON_VERSION}"
"$UV" venv --python "$PYTHON_VERSION"

echo "==> Installing dependencies (fastapi, uvicorn, pydantic, httpx, numpy, pytest, jupyterlab)"
"$UV" pip install --python .venv/bin/python \
  fastapi "uvicorn[standard]" pydantic httpx numpy pytest jupyterlab ipykernel nbclient nbformat

echo ""
echo "==> Installing this package in editable mode"
"$UV" pip install --python .venv/bin/python -e . --no-deps

echo ""
echo "==> Registering the Jupyter kernel (so the notebooks find this venv)"
# IMPORTANT: without this, Jupyter uses the system Python and the notebooks
# fail with "No module named 'fastapi'". This is the single most common
# setup failure -- the notebook opens, but the kernel is the wrong Python.
"$UV" pip install --python .venv/bin/python ipykernel
.venv/bin/python -m ipykernel install --user --name cc3po-ai-q3 \
    --display-name "CC3PO AI Q3" >/dev/null 2>&1 \
    && echo "    kernel 'cc3po-ai-q3' registered" \
    || echo "    WARNING: kernel registration failed (notebooks may not run)"

echo ""
echo "Done. Next steps:"
echo "  source .venv/bin/activate"
echo "  make test     # verify everything works -- NO API KEY NEEDED"
echo "  make serve    # start the FastAPI service on http://localhost:8000"
echo "  make lab      # open JupyterLab"
echo ""
echo "Optional (a real vector database):"
echo "  uv pip install --python .venv/bin/python -e '.[chroma]'"
