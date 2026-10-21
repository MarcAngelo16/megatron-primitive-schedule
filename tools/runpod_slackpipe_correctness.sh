#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export MAMBA_DETERMINISTIC=1 TRITON_CACHE_AUTOTUNING=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export SLACKPIPE_TEST_TRANSPORT="${SLACKPIPE_TEST_TRANSPORT:-nccl-p2p}"
exec "${PYTHON:-python}" -m tools.run_slackpipe_eval --phase correctness \
  --model-config configs/slackpipe_eval/nemotron_h_8b.json \
  --method slack-heuristic --pp 4 --num-stages 8 \
  --transport "$SLACKPIPE_TEST_TRANSPORT" "$@"
