#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m tools.run_slackpipe_eval --phase smoke \
  --model-config configs/slackpipe_eval/nemotron_h_8b.json \
  --method slack-heuristic --pp 4 --num-stages 8 "$@"
