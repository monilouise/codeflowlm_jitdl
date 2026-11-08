"""Shared settings of the RQ1 experiments (Section 4.4.1 of the paper)."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "rq1")
BASELINES_DIR = os.path.join(REPO_ROOT, "baselines")

# JIT-Defects4J projects used in RQ1 (commons-dbcp and commons-io are excluded, see Section 4.1).
PROJECTS = [
    "ant-ivy", "commons-bcel", "commons-beanutils", "commons-codec", "commons-collections",
    "commons-compress", "commons-configuration", "commons-digester", "commons-jcs", "commons-lang",
    "commons-math", "commons-net", "commons-scxml", "commons-validator", "commons-vfs", "giraph",
    "gora", "opennlp", "parquet-mr",
]

# Random seeds of the five repetitions of CodeFlowLM and BORB (33 is the default).
SEEDS = [33, 23, 99, 72, 22]

# CodeFlowLM configurations evaluated in the paper: name -> (pretrained_model, peft_alg).
CODEFLOWLM_CONFIGS = {
    "CodeT5+-LoRA": ("codet5p-770m", "lora"),
    "UniXCoder-PreT": ("unixcoder", "pret"),
    "UniXCoder-LoRA": ("unixcoder", "lora"),  # within-project only
}

# BORB base learners (jit-sdp-nn "--model" values).
BORB_MODELS = ["mlp", "ihf", "lr", "nb", "irf"]
BORB_MLP_EPOCHS = 80

RESULT_COLUMNS = ["method", "setting", "project", "seed", "gmean", "rdiff"]


def project_list(selected):
    """Expand ["all"] into PROJECTS and validate explicit project names."""
    if not selected or selected == ["all"]:
        return list(PROJECTS)
    unknown = sorted(set(selected) - set(PROJECTS))
    if unknown:
        raise SystemExit(f"Unknown project(s): {unknown}. Valid: {PROJECTS}")
    return selected
