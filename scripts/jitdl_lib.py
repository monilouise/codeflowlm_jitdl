"""Shared code of the JIT-DL experiments (RQ2 and RQ3).

Ported from jit-dl-llms/JIT-DL-GPT-5.ipynb without changing its logic:
  * the true-positive commits of CodeT5+ that are sent to the LLMs (Section 4.4.2);
  * the prompt of Listing 1;
  * parsing of the LLM answers (JSON, with a regex fallback for malformed JSON);
  * alignment of the dataset lines with the lines cited by the LLM;
  * the line-level metrics adapted from JIT-Fine (Top-k counts 1 when at least one
    defect-inducing line is among the first k lines, see footnote 1 of the paper).
"""
import json
import math
import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESPONSES_DIR = os.path.join(REPO_ROOT, "jit-dl-llms")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "rq2")

METRICS = ["top5", "top10", "recall20", "effort20", "ifa"]

# Prompt of Listing 1, kept verbatim (including non-breaking spaces) as sent to the LLMs.
with open(os.path.join(RESPONSES_DIR, "prompt_listing1.txt"), encoding="utf-8", newline="") as _f:
    PROMPT = _f.read()


# ---------------------------------------------------------------- data

def load_true_positives(data_dir=DATA_DIR):
    """Commits correctly predicted as defect-inducing by CodeT5+, with their diffs.
    The row order defines the order of the LLM responses."""
    preds = pd.read_csv(os.path.join(data_dir, "results.csv"))
    tp = preds[(preds["true_label"] == 1) & (preds["pred_label"] == 1)]
    tp = tp.rename(columns={"commit_id": "commit_hash"})
    features = pd.read_pickle(os.path.join(data_dir, "features_test_with_diffs.pkl"))
    return pd.merge(tp, features, on="commit_hash")


def load_line_labels(data_dir=DATA_DIR):
    """Line-level labels of JIT-Defects4J (one row per changed line)."""
    return pd.read_pickle(os.path.join(data_dir, "changes_complete_buggy_line_level.pkl"))


def load_buggy_changes(data_dir=DATA_DIR):
    """Original (unnormalized) changed lines: {project: {commit: {"added": {file: [lines]}, "deleted": ...}}}."""
    with open(os.path.join(data_dir, "buggy_changes_with_buggy_line.json"), encoding="utf-8") as f:
        by_project = json.load(f)
    return {commit: changes for commits in by_project.values() for commit, changes in commits.items()}


def build_prompt(diff):
    return sanitize_text(PROMPT + "\n<diff>\n" + diff + "\n</diff>")


def sanitize_text(text, replace_with="�"):
    """Replace invalid surrogates so the prompt is valid UTF-8."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"[\ud800-\udfff]", replace_with, text)
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    return text.encode("utf-8", "replace").decode("utf-8")


# ---------------------------------------------------------------- parsing

def _escape_codecontent_quotes(text):
    def repl(m):
        content = re.sub(r'(?<!\\)"', r'\\"', m.group(1))
        return f'"codeContent": "{content}"'

    return re.sub(r'"codeContent":\s*"([^"]*?)"', repl, text)


def _parse_with_regex(raw):
    """Fallback for answers that are not valid JSON (e.g. unescaped quotes inside codeContent)."""
    intent = re.search(r'"intentOfThisCommit"\s*:\s*"([^"]*)"', raw)
    faults = []
    for item in re.findall(r'\{\s*"lineNumber"[\s\S]*?"score"\s*:\s*\d+\s*\}', raw):
        cc = re.search(r'"codeContent"\s*:\s*"(.*)"\s*,', item, flags=re.S)
        code = cc.group(1) if cc else ""
        code = "".join('\\"' if c == '"' and (i == 0 or code[i - 1] != "\\") else c for i, c in enumerate(code))
        reason = re.search(r'"reason"\s*:\s*"(.*)"\s*,\s*"score"', item, flags=re.S)
        faults.append({
            "lineNumber": int(re.search(r'"lineNumber"\s*:\s*(\d+)', item).group(1)),
            "codeContent": code,
            "reason": reason.group(1).strip() if reason else "",
            "score": int(re.search(r'"score"\s*:\s*(\d+)', item).group(1)),
        })
    return {"intentOfThisCommit": intent.group(1) if intent else None, "faultLocalization": faults}


def parse_response(response):
    """Return the list of faults of an LLM answer, sorted by decreasing score."""
    if isinstance(response, str):
        try:
            begin, end = response.index("{"), response.rfind("}")
        except ValueError:
            begin, end = 0, len(response)
        response = response[begin:end + 1]
        try:
            response = json.loads(_escape_codecontent_quotes(response))
        except json.JSONDecodeError:
            response = _parse_with_regex(response)
    faults = response["faultLocalization"]
    if faults and "score" in faults[0]:
        faults = sorted(faults, key=lambda x: x["score"], reverse=True)
    return faults


# ---------------------------------------------------------------- line alignment

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _tokenized_added_lines(changes):
    return [(line, set(_TOKEN_RE.findall(line))) for lines in changes["added"].values() for line in lines]


def find_original_line(changes, raw_changed_line, tokenized=None):
    """Map a normalized dataset line to the original added line sharing most tokens.
    As in the original evaluation, only added lines are searched."""
    raw_tokens = raw_changed_line.split()
    best_match, best_line = 0, None
    for line, line_tokens in tokenized or _tokenized_added_lines(changes):
        match = sum(1 for token in raw_tokens if token in line_tokens)
        if match > best_match:
            best_match, best_line = match, line
    return best_line


def find_in_faults(original_line, faults):
    for fault in faults:
        if original_line and original_line in fault["codeContent"]:
            return fault
    return None


def align_commit_lines(lines, changes):
    """Copy of the labelled lines of a commit with the matching original line (column original_line)."""
    tokenized = _tokenized_added_lines(changes)
    lines = lines.copy()
    lines["original_line"] = [find_original_line(changes, raw, tokenized) for raw in lines["raw_changed_line"]]
    return lines


def score_commit_lines(faults, aligned_lines):
    """Score each line with the score of the fault that cites it (0 otherwise); keeps the matched fault."""
    lines = aligned_lines.copy()
    matched = [find_in_faults(orig, faults) for orig in lines["original_line"]]
    lines["fault"] = matched
    lines["score"] = [f["score"] if f else 0 for f in matched]
    return lines


_ALIGNMENT_CACHE = {}


def scored_commits(responses, tp, line_labels, buggy_changes):
    """Yield (index, true-positive row, scored lines) for every commit whose answer cites at least one line."""
    for i, response in enumerate(responses):
        faults = parse_response(response)
        if len(faults) == 0:
            continue
        commit_id = tp.loc[i, "commit_hash"]
        if commit_id not in _ALIGNMENT_CACHE:  # the alignment does not depend on the answer
            lines = line_labels[line_labels["commit_id"] == commit_id]
            _ALIGNMENT_CACHE[commit_id] = align_commit_lines(lines, buggy_changes[commit_id])
        yield i, tp.loc[i], score_commit_lines(faults, _ALIGNMENT_CACHE[commit_id])


# ---------------------------------------------------------------- metrics

def line_level_metrics(line_score, label):
    """IFA, Recall@20%, Effort@20%, Top-10 and Top-5 of one commit (adapted from JIT-Fine)."""
    line_score = MinMaxScaler().fit_transform(np.array(line_score, dtype=float).reshape(-1, 1))
    line_df = pd.DataFrame({"scr": line_score.ravel(), "label": label})
    line_df = line_df.sort_values(by="scr", ascending=False)
    line_df["row"] = np.arange(1, len(line_df) + 1)
    real_buggy_lines = line_df[line_df["label"] == 1]

    if len(real_buggy_lines) < 1:
        return {"ifa": len(line_df), "recall20": 0, "effort20": math.ceil(0.2 * len(line_df)),
                "top10": 0, "top5": 0}

    labels = list(line_df["label"])
    top20 = line_df.head(int(0.2 * len(line_df)))
    buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
    return {
        "ifa": real_buggy_lines.iloc[0]["row"] - 1,
        "recall20": len(top20[top20["label"] == 1]) / len(real_buggy_lines),
        "effort20": int(buggy_20_percent.iloc[-1]["row"]) / len(line_df),
        # 1 if at least one defect-inducing line is ranked within the first k lines
        "top10": 1.0 if np.sum(labels[:10]) > 0 else 0.0,
        "top5": 1.0 if np.sum(labels[:5]) > 0 else 0.0,
    }


def evaluate_responses(responses, tp, line_labels, buggy_changes):
    """Mean of each metric over the true-positive commits of one run."""
    rows = [line_level_metrics(lines["score"].tolist(), lines["label"].tolist())
            for _, _, lines in scored_commits(responses, tp, line_labels, buggy_changes)]
    df = pd.DataFrame(rows)
    return {m: float(df[m].mean()) for m in METRICS} | {"n_commits": len(df)}
