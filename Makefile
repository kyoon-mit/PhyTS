.PHONY: help env env-fm env-jax

WORKDIR := $(shell pwd)
FM_DIR  := $(WORKDIR)/benchmarks/foundation

help:
	@echo ""
	@echo "Environment setup"
	@echo "-----------------"
	@echo "  make env      Python 3.12 venv for main repo (torch, lightning, JAX optional)"
	@echo "  make env-jax  Same as env but with JAX + CUDA 12 extras"
	@echo "  make env-fm   Python 3.10 venv for foundation-model benchmarks"
	@echo ""
	@echo "After setup, activate with:"
	@echo "  source .venv/bin/activate               # main"
	@echo "  source benchmarks/foundation/.venv/bin/activate  # foundation models"
	@echo ""

env:
	uv sync

env-jax:
	uv sync --extra jax --extra cu12

env-fm:
	cd $(FM_DIR) && uv sync
