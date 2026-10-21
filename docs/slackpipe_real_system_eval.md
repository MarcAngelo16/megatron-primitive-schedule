# Real-system Evaluation

The current implementation has one staged entry point,
`tools/run_slackpipe_eval.py`, and one five-method registry,
`tools/slackpipe_eval_config.py`. A thin campaign rotates methods and repeats
the same workflow; it does not reimplement calibration, solver selection,
profile integrity, backend selection or receipts.

See [Five-method Evaluation Contract](slackpipe_five_methods.md) for executable
commands, topology, model/cost/plan compatibility, calibrated-cost sharing,
thread and budget controls, schema definitions, plotting and tests.

Shared implementation:

| Module | Responsibility |
| --- | --- |
| `slackpipe_eval_args.py` | Parser and normalization, used by launcher and worker |
| `slackpipe_eval_common.py` | Atomic JSON, source/artifact hashes, process timeout cleanup, run statistics |
| `slackpipe_training.py` | Shared training step, construction plans, initialization and calibration partitions |
| `slackpipe_model_support.py` | Model/batch/parameter helpers shared with correctness tests |
| `slackpipe_eval_worker.py` | Target-model adapter and profile/plan provenance |
| `slackpipe_profiling.py` | Sequential stage/layer collection, explicit selection, partial outcomes; no retries |
| `slackpipe_eval_receipts.py` | v4 phase identity, integrity, archive/resume |
| `slackpipe_eval_tables.py` | Registry-driven comparisons from accepted receipts |

None of the normal experiment modules imports a test implementation or a
retired model-specific launcher. Manifest/fit remains available via
`tools/slackpipe_hybrid.py`; the redundant specialized execution commands were
removed. Earlier full experiment receipts/results remain historical and must
be interpreted using their recorded source revision, not the current contracts.

See [Profiling contract](slackpipe_profiling.md) for `--phase`, `--num-stages`,
`--profiling-method`, `--solver-profile` and explicit artifact examples. Benchmark
summary v2 computes main statistics and the normal 95% confidence half-width
from independent continuous wall-time run observations. CUDA-event summaries
are separate; a single run has no confidence interval.
