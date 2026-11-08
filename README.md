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
(exported with `scripts/rq1_borb.py export` from the original MLflow folders), and `results/rq1/paper/<wp|cp>.csv`
contains the per-project means of Tables 11 and 12; `make rq1-stats-paper` reproduces every statistic reported in
Section 5.1 from them. In commons-lang (CP), the CodeT5+ and UniXCoder-PreT G-means are stored with five decimals
(0.67492 / 0.67496), as explained in the note of Table 12.

**Environments.**
- *CodeFlowLM*: A100 or another GPU with at least 24 GB of VRAM for CodeT5+ with batch size 16 (a T4 is enough for
  UniXCoder). Tested on Google Colab. `make setup-rq1` installs `envs/requirements-rq1-codeflowlm.txt`.
  Each run is limited to 10 hours by codeflowlm; rerun the same command to resume from the last training step.
- *BORB*: [monilouise/jit-sdp-nn](https://github.com/monilouise/jit-sdp-nn) (commit `eb286ee`), a fork of
  [dinaldoap/jit-sdp-nn](https://github.com/dinaldoap/jit-sdp-nn) adapted to JIT-Defects4J, in `baselines/jit-sdp-nn`.
  Install it with `conda env create --name borb --file conda.yml` and `pip install -r requirements.lock` inside that folder.
  Default BORB hyperparameters are used (MLP with 80 epochs).
- *HumLa / Eco-HumLa*: [HumLa-fse2023](https://github.com/sunnysong14/HumLa-fse2023) adapted to JIT-Defects4J and to
  the CP setting, in `baselines/HumLa-fse2023` (Python 3.9, numpy, scikit-learn, scikit-multiflow 0.5.3).
  The parameters selected by grid search over the first 50 commits of each project are in
  `codes/data_stream/para_bst_30sd.csv`.

Seeds: 33 (default), 23, 99, 72 and 22 for CodeFlowLM and BORB; HumLa uses its own seeds 0-4.

The original Colab notebooks are still available:
[continual_jitsdp/CodeFlowLM-WP.ipynb](continual_jitsdp/CodeFlowLM-WP.ipynb) and
[continual_jitsdp/CodeFlowLM-CP.ipynb](continual_jitsdp/CodeFlowLM-CP.ipynb) (one seed per notebook).

#### Inference time measurements

[continual_jitsdp/PredictionTimesTest.ipynb](continual_jitsdp/PredictionTimesTest.ipynb): tests CodeFlowLM inference time for a single prediction with both base learners (CodeT5+ and UniXCoder).

### Defect Localization (RQs 2 and 3)

- [jit-dl-llms/JIT-DL-GPT-5.ipynb](jit-dl-llms/JIT-DL-GPT-5.ipynb): Quantitative and qualitative experiments with GPT-5, the LLM with the best observed metrics in our experiments.
- response-gpt-5_exec-*.pkl files: Different runs for the same LLM

#### False-positives analysis

- Prompt: "I am attaching a .json file containing a list of false positives, i.e., lines classified as defective when they are actually non-defective. Analyze the file and return a list of the most common types of false positives, explicitly indicating the corresponding JSON excerpts and their respective projects."
- File: [jit-dl-llms/fps.json](jit-dl-llms/fps.json) 

#### False-negatives analysis

- Prompt: "The attached .txt file contains a list of false negatives, i.e., lines classified as clean when they are actually defective. Analyze the file and return a list of the most common types of false negatives."
- File: [jit-dl-llms/fns.txt](jit-dl-llms/fns.txt) 
   
