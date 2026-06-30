.PHONY: install test baseline compress eval visualize report repro clean

PYTHON := python
CHI ?= 64

install:
	pip install -e .[dev]

test:
	pytest tests/ -v --tb=short

baseline:
	$(PYTHON) scripts/run_baseline.py

compress:
	$(PYTHON) scripts/compress_model.py --chi 16
	$(PYTHON) scripts/compress_model.py --chi 32
	$(PYTHON) scripts/compress_model.py --chi 64
	$(PYTHON) scripts/compress_model.py --chi 128

compress-one:
	$(PYTHON) scripts/compress_model.py --chi $(CHI)

eval:
	$(PYTHON) scripts/run_eval.py

visualize:
	$(PYTHON) scripts/visualize.py --chi $(CHI)

report:
	cd docs/report && pdflatex report.tex && pdflatex report.tex

repro:
	@echo "Running full end-to-end reproducibility test..."
	$(PYTHON) scripts/run_baseline.py
	$(PYTHON) scripts/compress_model.py --chi 64
	$(PYTHON) scripts/run_eval.py --chi 64
	$(PYTHON) scripts/visualize.py --chi 64
	@echo "Reproducibility test complete. Check results/eval_summary.json"

smoke:
	$(PYTHON) -c "import vlam_compress; print('vlam_compress: ok')"
	$(PYTHON) -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
	$(PYTHON) -c "import quimb; print('quimb: ok')"
	$(PYTHON) -c "import bitsandbytes; print('bitsandbytes: ok')"
	$(PYTHON) -c "import mujoco; print('mujoco: ok')"
	$(PYTHON) -c "import pennylane; print('pennylane:', pennylane.__version__)"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
