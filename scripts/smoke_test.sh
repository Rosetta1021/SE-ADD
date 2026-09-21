#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
"$PYTHON_BIN" -m compileall -q "$REPO_ROOT/seadd" "$REPO_ROOT/backends" "$REPO_ROOT/tests"
"$PYTHON_BIN" "$REPO_ROOT/tests/test_schema_and_builder.py"
"$PYTHON_BIN" "$REPO_ROOT/backends/qwen2audio/generate_hints.py" --help >/dev/null
"$PYTHON_BIN" "$REPO_ROOT/backends/qwen2audio/score_logits.py" --help >/dev/null
"$PYTHON_BIN" "$REPO_ROOT/backends/qwen2audio/train_initial_lora.py" --help >/dev/null
"$PYTHON_BIN" "$REPO_ROOT/backends/qwen2audio/train_cumulative_lora.py" --help >/dev/null
echo "PASS: compile, schema/builder, and Qwen CLI import smoke tests"
