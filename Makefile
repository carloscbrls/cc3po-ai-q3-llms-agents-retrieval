# Makefile for the Q3 scaffold repo.
#
#     make setup   -- create .venv and install deps
#     make test    -- run the test suite (NO API KEY REQUIRED)
#     make lab     -- start JupyterLab
#     make serve   -- start the FastAPI service
#     make check   -- syntax-check every .py file
#     make notebooks -- execute every notebook (CI use)

PY ?= .venv/bin/python
UV ?= uv
PIP_INSTALL ?= $(UV) pip install --python $(PY)

.PHONY: help setup test lab serve check notebooks corpus clean

help:
	@echo "CC3PO Q3 scaffold -- LLMs, agents, and retrieval"
	@echo ""
	@echo "  make setup      create .venv and install deps"
	@echo "  make test       run pytest (no API key needed)"
	@echo "  make lab        start JupyterLab at http://localhost:8888"
	@echo "  make serve      start the FastAPI service at http://localhost:8000"
	@echo "  make notebooks  execute every notebook and report pass/fail"
	@echo "  make check      syntax-check every .py file"
	@echo "  make clean      remove caches"

setup:
	$(UV) venv --python 3.12
	$(PIP_INSTALL) fastapi "uvicorn[standard]" pydantic httpx numpy pytest jupyterlab ipykernel nbclient nbformat
	$(PIP_INSTALL) -e . --no-deps
	@echo "==> Registering Jupyter kernel"
	$(PY) -m ipykernel install --user --name cc3po-ai-q3 --display-name "CC3PO AI Q3" >/dev/null 2>&1 \
		&& echo "    kernel 'cc3po-ai-q3' registered" \
		|| echo "    WARNING: kernel registration failed"

test:
	$(PY) -m pytest tests -v

lab:
	$(PY) -m jupyterlab --notebook-dir notebooks

serve:
	$(PY) -m uvicorn src.service:app --reload --port 8000

notebooks:
	$(PY) scripts/run_notebooks.py

corpus:
	$(PY) scripts/build_corpus.py

check:
	@$(PY) -c "import ast,pathlib,sys; \
	files=[f for f in pathlib.Path('.').rglob('*.py') if '.venv' not in f.parts]; \
	[ast.parse(f.read_text(), filename=str(f)) for f in files]; \
	print(f'syntax OK: {len(files)} python files'); \
	sys.exit(0)"

clean:
	rm -rf .pytest_cache **/__pycache__ .ipynb_checkpoints
	find . -name '__pycache__' -type d -not -path './.venv/*' -exec rm -rf {} +
	find . -name '*.pyc' -not -path './.venv/*' -delete
