# Single-command entry points of the replication package.
# Each component runs in its own Python environment (see envs/ and README.md):
#   PYTHON        CodeFlowLM (GPU) and the analysis scripts
#   BORB_PYTHON   environment where the jit-sdp-nn fork is installed (provides `jitsdp`)
#   HUMLA_PYTHON  environment of HumLa (Python 3.9 + scikit-multiflow)
# Example: make rq1 HUMLA_PYTHON=~/miniconda3/envs/humla/bin/python

PYTHON ?= python
# On Windows, plain "bash" may resolve to WSL; point it to Git Bash instead, e.g.
#   make setup-rq1 BASH="C:/Program Files/Git/bin/bash.exe"
BASH ?= bash
BORB_PYTHON ?= $(PYTHON)
HUMLA_PYTHON ?= $(PYTHON)

.PHONY: setup-rq1 rq1 rq1-codeflowlm rq1-borb rq1-humla rq1-stats rq1-stats-paper

setup-rq1:
	"$(BASH)" scripts/setup_rq1.sh

# Everything for RQ1: CodeFlowLM, the three baselines, then the statistical tests.
rq1: rq1-codeflowlm rq1-borb rq1-humla rq1-stats

rq1-codeflowlm:
	$(PYTHON) scripts/rq1_codeflowlm.py --setting wp --config all
	$(PYTHON) scripts/rq1_codeflowlm.py --setting cp --config all

rq1-borb:
	$(BORB_PYTHON) scripts/rq1_borb.py run --setting wp
	$(BORB_PYTHON) scripts/rq1_borb.py run --setting cp

rq1-humla:
	$(HUMLA_PYTHON) scripts/rq1_humla.py --setting wp
	$(HUMLA_PYTHON) scripts/rq1_humla.py --setting cp

# Friedman / Iman-Davenport / Nemenyi / Kendall's W / Cliff's delta over the runs in results/rq1/runs/.
rq1-stats:
	$(PYTHON) scripts/rq1_stats.py "results/rq1/runs/wp/*.csv" --setting wp --cd-plot results/rq1/cd_wp.png
	$(PYTHON) scripts/rq1_stats.py "results/rq1/runs/cp/*.csv" --setting cp --cd-plot results/rq1/cd_cp.png

# Same tests over the per-project means published in Tables 11 and 12 (no training needed).
rq1-stats-paper:
	$(PYTHON) scripts/rq1_stats.py results/rq1/paper/wp.csv
	$(PYTHON) scripts/rq1_stats.py results/rq1/paper/cp.csv

# ---------------------------------------------------------------- RQ2
.PHONY: rq2 rq2-llm rq2-metrics rq2-attention

# Table 5 (LLM rows) from the answers stored in jit-dl-llms/ (no API key needed).
rq2: rq2-metrics

rq2-metrics:
	$(PYTHON) scripts/rq2_metrics.py

# Query the LLMs again (API keys required; answers go to results/rq2/responses/), then evaluate them.
rq2-llm:
	$(PYTHON) scripts/rq2_llm.py --model gpt-5 --runs 5
	$(PYTHON) scripts/rq2_llm.py --model gemini
	$(PYTHON) scripts/rq2_llm.py --model claude
	$(PYTHON) scripts/rq2_metrics.py --responses-dir results/rq2/responses --out-dir results/rq2/rerun

# Attention-based baselines CodeT5+ and UniXCoder (GPU; needs `make setup-rq1` for deps/PEFT4CC).
# Results go to results/rq2/baselines/ and are included in the next `make rq2`.
rq2-attention:
	$(PYTHON) scripts/rq2_attention.py --config CodeT5+-LoRA
	$(PYTHON) scripts/rq2_attention.py --config UniXCoder-PreT
