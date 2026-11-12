# CodeFlowLM: Incremental Just-In-Time Defect Prediction with Pretrained Language Models and Exploratory Insights into Defect Localization

## Replication steps

### RQ1: Continual JIT-SDP with PLMs

| Step | Command | Environment | Output |
|---|---|---|---|
| Fetch dependencies (codeflowlm, PEFT4CC, pinned commits) | `make setup-rq1` | CodeFlowLM | `deps/` |
| CodeFlowLM, 3 configurations x 19 projects x 5 seeds (WP and CP) | `make rq1-codeflowlm` | CodeFlowLM, GPU | `results/rq1/runs/<wp\|cp>/codeflowlm.csv` |
| BORB, 5 base learners x 19 projects x 5 seeds (WP and CP) | `make rq1-borb BORB_PYTHON=...` | jit-sdp-nn | `results/rq1/runs/<wp\|cp>/borb.csv` |
| HumLa and Eco-HumLa, 19 projects x 5 seeds (WP and CP) | `make rq1-humla HUMLA_PYTHON=...` | HumLa | `results/rq1/runs/<wp\|cp>/humla.csv` |
| Friedman, Iman-Davenport, Nemenyi, Kendall's W, Cliff's delta (Section 5.1) | `make rq1-stats` | any (pandas, scipy) | console + `results/rq1/cd_<wp\|cp>.png` |
| Same tests over the means published in Tables 11 and 12 | `make rq1-stats-paper` | any (pandas, scipy) | console |

`make rq1` runs all of the above in sequence. Every script accepts `--projects`, `--seeds` and `--dry-run`
for partial or quick runs (e.g. `python scripts/rq1_codeflowlm.py --setting wp --config CodeT5+-LoRA --projects gora --seeds 33`),
and skips combinations whose results already exist, so an interrupted run is resumed by repeating the command.
Each run writes one row per (method, project, seed) with the prequential G-mean and |R1 - R0| (fading factor 0.99).

**Results already included.** `results/rq1/runs/<wp|cp>/borb.csv` contains the BORB runs used in the paper
(exported with `scripts/rq1_borb.py export` from the original MLflow folders), `results/rq1/runs/<wp|cp>/humla.csv`
the HumLa and Eco-HumLa runs produced by `scripts/rq1_humla.py`, and `results/rq1/paper/<wp|cp>.csv`
contains the per-project means of Tables 11 and 12; `make rq1-stats-paper` reproduces every statistic reported in
Section 5.1 from them. In commons-lang (CP), the CodeT5+ and UniXCoder-PreT G-means are stored with five decimals
(0.67492 / 0.67496), as explained in the note of Table 12.

**Environments.**
- *CodeFlowLM*: A100 or another GPU with at least 24 GB of VRAM for CodeT5+ with batch size 16 (a T4 is enough for
  UniXCoder). Tested on Google Colab. `make setup-rq1` installs `envs/requirements-rq1-codeflowlm.txt`.
  Each run is limited to 10 hours by codeflowlm; rerun the same command to resume from the last training step.
- *BORB*: the code is included in `baselines/jit-sdp-nn`, a copy of
  [dinaldoap/jit-sdp-nn](https://github.com/dinaldoap/jit-sdp-nn) adapted to JIT-Defects4J.
  Default BORB hyperparameters are used (MLP with 80 epochs). Create the environment inside that folder:
  ```bash
  conda env create --name borb --file conda.yml
  conda run --name borb python -m pip install "pip==24.0" "setuptools<68" "wheel==0.37.1"
  conda run --name borb pip install -r requirements.lock
  conda run --name borb pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  conda run --name borb jitsdp --help
  ```
  The second line is needed because the pip shipped with Python 3.7 would otherwise fetch a `setuptools` that requires
  Python 3.8 (if the editable install of `jitsdp` still fails, add `--no-build-isolation` to the third line).
  `torch` is commented out in `requirements.lock` but is required (MLP); if the wheel above is not found, use
  `conda install --name borb pytorch=1.4.0 cpuonly -c pytorch`. Then run, from the repository root,
  `make rq1-borb BORB_PYTHON=<conda envs folder>/borb/python.exe` (`.../borb/bin/python` on Linux/macOS);
  the environment does not need to be activated.
- *HumLa / Eco-HumLa*: the code is included in `baselines/HumLa-fse2023`, a copy of
  [HumLa-fse2023](https://github.com/sunnysong14/HumLa-fse2023) adapted to JIT-Defects4J and to
  the CP setting (Python 3.9, numpy, scikit-learn, scikit-multiflow 0.5.3).
  The parameters selected by grid search over the first 50 commits of each project are in
  `codes/data_stream/para_bst_30sd.csv`.

Seeds: 33 (default), 23, 99, 72 and 22 for CodeFlowLM and BORB; HumLa uses its own seeds 0-4.

The original Colab notebooks are still available:
[continual_jitsdp/CodeFlowLM-WP.ipynb](continual_jitsdp/CodeFlowLM-WP.ipynb) and
[continual_jitsdp/CodeFlowLM-CP.ipynb](continual_jitsdp/CodeFlowLM-CP.ipynb) (one seed per notebook).

#### Inference time measurements

[continual_jitsdp/PredictionTimesTest.ipynb](continual_jitsdp/PredictionTimesTest.ipynb): tests CodeFlowLM inference time for a single prediction with both base learners (CodeT5+ and UniXCoder).

### RQ2: LLMs for JIT-DL

| Step | Command | Requirements | Output |
|---|---|---|---|
| LLM rows of Table 5 from the stored answers | `make rq2` | pandas, scikit-learn | console + `results/rq2/llm_runs.csv`, `results/rq2/summary.csv` |
| Query GPT-5 (5 runs), Gemini 2.5 Pro and Claude Sonnet 4.5 again, then evaluate | `make rq2-llm` | `openai`, `google-genai`, `anthropic`; API keys | `results/rq2/responses/`, `results/rq2/rerun/` |
| Attention-based baselines CodeT5+ and UniXCoder, 5 seeds (PEFT4CC `--do_locate_defects`) | `make rq2-attention` | GPU, `make setup-rq1` | `results/rq2/baselines/*.csv` (added to `results/rq2/summary.csv` by `make rq2`) |

The published values of Table 5 are in `results/rq2/paper/table5.csv`.

The LLMs receive the prompt of Listing 1 ([jit-dl-llms/prompt_listing1.txt](jit-dl-llms/prompt_listing1.txt), kept
verbatim) followed by the diff of each of the 221 commits correctly predicted as defect-inducing by CodeT5+
(`data/results.csv`). The answers used in the paper are stored in `jit-dl-llms/responses-*.pkl`
(`responses-gpt-5_exec-0..4.pkl` are the five GPT-5 runs). [scripts/jitdl_lib.py](scripts/jitdl_lib.py) contains
the answer parsing, the alignment between dataset lines and the lines cited by the LLM, and the line-level metrics,
ported from [jit-dl-llms/JIT-DL-GPT-5.ipynb](jit-dl-llms/JIT-DL-GPT-5.ipynb) without changes in logic.
Use `python scripts/rq2_llm.py --model <gpt-5|claude|gemini> --limit 3` for a quick check of the API calls, and
`--dry-run` to print the first prompt.

Notes:
- LLM answers are not deterministic, so new queries will give slightly different values.
- Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) is scheduled for retirement by Anthropic on 2026-11-30; after that,
  its row can only be reproduced from the stored answers.
- IFA depends on the order of tied lines (most lines of a commit get score 0), and the sorting of ties may vary
  with the numpy version and CPU. With numpy 2.4 the stored answers reproduce the published Top-5, Recall@20% and
  Effort@20% of every model (Top-10 within 0.001); the mean IFA is up to 0.05 lower, because of the tie order of
  commit `94faa31` (commons-lang, 312 changed lines, 308 of them with score 0).

#### False-positives analysis

- Prompt: "I am attaching a .json file containing a list of false positives, i.e., lines classified as defective when they are actually non-defective. Analyze the file and return a list of the most common types of false positives, explicitly indicating the corresponding JSON excerpts and their respective projects."
- File: [jit-dl-llms/fps.json](jit-dl-llms/fps.json) 

#### False-negatives analysis

- Prompt: "The attached .txt file contains a list of false negatives, i.e., lines classified as clean when they are actually defective. Analyze the file and return a list of the most common types of false negatives."
- File: [jit-dl-llms/fns.txt](jit-dl-llms/fns.txt) 
   
