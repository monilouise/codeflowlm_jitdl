#!/usr/bin/env bash
# Fetches the code RQ1 depends on, pinned to the commits used in the paper.
#   deps/codeflowlm   - CodeFlowLM training loop (github.com/monilouise/codeflowlm)
#   deps/PEFT4CC      - base learners fine-tuned with LoRA / prefix tuning (github.com/monilouise/PEFT4CC)
# Usage: bash scripts/setup_rq1.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPS="$ROOT/deps" 
mkdir -p "$DEPS"

clone_pinned() {
    local url=$1 dir=$2 commit=$3
    if [ ! -d "$DEPS/$dir/.git" ]; then
        git clone "$url" "$DEPS/$dir"
    fi
    git -C "$DEPS/$dir" fetch --quiet origin
    git -C "$DEPS/$dir" checkout --quiet "$commit"
    echo "$dir @ $(git -C "$DEPS/$dir" rev-parse --short HEAD)"
}

clone_pinned https://github.com/monilouise/codeflowlm.git codeflowlm 2b86a06
clone_pinned https://github.com/monilouise/PEFT4CC.git PEFT4CC c86785b

pip install -r "$ROOT/envs/requirements-rq1-codeflowlm.txt"
