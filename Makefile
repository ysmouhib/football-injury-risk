# Task runner. Every target delegates to the Python CLI (python -m src.cli <task>),
# which is the supported entry point on machines without make.
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= .venv/bin/python
endif

.PHONY: data features test train evaluate report all

data:
	$(PY) -m src.cli data

features:
	$(PY) -m src.cli features

test:
	$(PY) -m src.cli test

# usage: make train EXP=configs/exp-0001.yaml
train:
	$(PY) -m src.cli train $(EXP)

evaluate:
	$(PY) -m src.cli evaluate $(EXP)

report:
	$(PY) -m src.cli report

all: data features test
