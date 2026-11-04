.PHONY: test lint audit
PYTHON ?= python

test:
	CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests $(PYTHON) -m pytest -q tests
lint:
	$(PYTHON) tools/lint_standing_rules.py
audit:
	$(PYTHON) tools/audit_instruction_layers.py
