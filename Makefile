# AXEN Makefile
# Cross-platform commands for managing, testing, and running AXEN

PYTHON ?= python
PYTEST ?= pytest

.PHONY: all help setup check-hardware test run clean

all: help

help:
	@echo "=========================================================="
	@echo "  AXEN AI ASSISTANT & BRAIN — COMMAND TARGETS"
	@echo "=========================================================="
	@echo "  make setup          Install minimal Python dependencies"
	@echo "  make check-hardware Run hardware detection and sizing analysis"
	@echo "  make test           Run automated pytest test suite"
	@echo "  make run            Start AXEN main interface"
	@echo "  make clean          Remove Python cache and temporary files"
	@echo "=========================================================="

setup:
	$(PYTHON) -m pip install -r requirements.txt

check-hardware:
	$(PYTHON) scripts/hardware_check.py

test:
	$(PYTHON) -m pytest tests/ -v

run:
	$(PYTHON) main.py

clean:
	@echo "Cleaning cache files..."
	-rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
