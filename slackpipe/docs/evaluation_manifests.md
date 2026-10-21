> Historical scope: this document records pre-five-method code/results. Commands and method IDs below are not current entry points. Use the [current workflow](../../docs/slackpipe_five_methods.md); historical results keep their recorded commit and labels.

# Evaluation Manifests

SlackPipe CAL experiments are declared before execution as JSONL manifests. Each
line is exactly one method run. The runner reads those rows, constructs the
matching `slackpipe_cli` command, resumes completed valid rows, and stamps
runner provenance into the top-level JSON while leaving `canonical_result`
unchanged.

## Row Schema

`manifest_schema_version = 1` rows include:

- identity: `manifest_id`, `manifest_hash`, `run_id`, `experiment_group`,
  `configuration_id`, `repetition_id`
- method contract: `method`, `canonical_method_expected`,
  `method_contract_hash_expected`
- controlled problem fields: `B`, `N`, `W`, `L`, `mapping_type`,
  `stage_to_worker_mapping`, forward/backward ratios, communication profile and
  scalar `communication_ticks`
- activation fields: model, units, optional explicit stage units, cap mode,
  materialized cap vector, cap source, cap derivation hash, and enforcement flag
- solver fields: fixed-order partition backend, time limit, seed, solver
  threads, alternating max rounds
- output fields: `output_json`, `stdout_log`, `stderr_log`
- expected schema/provenance versions and git dirty policy

The companion `*.metadata.json` records generator time, generator/build git
provenance, host info, matrix definition, row count, method/config/seed lists,
default time limits, OR-Tools requirement, and the smoke/production flag.

`manifest_hash` is SHA-256 over normalized rows plus controlled metadata,
excluding every `manifest_hash` field and volatile host/time fields. Changing a
controlled field changes the hash; regenerating the same profile in the same
source/build contract does not.

## Profiles

Smoke:

- one tiny configuration: `B=2, N=2, W=2, L=4`
- communication profile `none`, scalar ticks `0`
- six canonical methods
- uncapped plus equal-memory rows
- one seed for every method
- total rows: `12`

Smoke with 1F1B:

- profile name: `smoke_big_1f1b`
- one tiny cyclic-interleaved configuration: `B=4, N=4, W=2, L=16`
- seven methods, including both deterministic fixed-order baselines
- uncapped plus equal-memory rows
- one seed for every method
- total rows: `14`

Main:

- `W in {2, 4, 8}`
- `stage_multiplier in {1, 2, 4}`, `N = W * stage_multiplier`
- `batch_ratio in {0.5, 1, 2}`, `B = int(N * batch_ratio)`
- `L = 8 * N`
- mapping type: cyclic stage mod worker
- forward/backward ratio: `1:2`
- communication profiles:
  - `none = 0` ticks
  - `moderate = 2` ticks, 25% of an average 8-tick forward stage
  - `heavy = 8` ticks, 100% of an average 8-tick forward stage
- activation modes:
  - uncapped diagnostic
  - equal-memory with materialized uniform-baseline caps
- row count formula:
  `81 configurations * 2 activation modes * (1 deterministic + 5 solver-backed * 3 seeds) = 2592`

Main with 1F1B:

- profile name: `main_big_5min_1f1b`
- same matrix as Main
- methods: `uniform-breadth-first`, `uniform-interleaved-1f1b`, and the five
  solver-backed canonical methods
- activation modes: uncapped plus equal-memory with materialized
  uniform-baseline caps
- default row budget: `300` seconds
- default solver threads: `8`
- fixed-order partition backend: `cpsat`
- row count formula:
  `81 configurations * 2 activation modes * (2 deterministic + 5 solver-backed * 3 seeds) = 2754`

Uncapped Main with 1F1B:

- profile name: `main_big_5min_1f1b_uncapped`
- same matrix and methods as `main_big_5min_1f1b`
- activation mode: uncapped only
- every row has `activation_cap_mode = none` and
  `enforce_activation_cap = false`
- no row contains `activation_cap_units_per_worker` or
  `activation_cap_derivation_hash`
- default row budget: `300` seconds
- default solver threads: `8`
- fixed-order partition backend: `cpsat`
- row count formula:
  `81 configurations * (2 deterministic + 5 solver-backed * 3 seeds) = 1377`

Ablation:

- three representative configurations: small, medium, larger
- same canonical contract-hashed CAL methods
- no invented SlackPipe wrapper rows unless those methods have stable
  `describe-method` contracts
- total rows with default seeds: `96`

Oracle:

- tiny configurations only
- separate `partition-only-fixed-order` rows with backend `enumerate`
- no hidden oracle objective is computed inside baseline rows
- total rows with default generator settings: `4`

`--profile all` concatenates smoke, main, ablation, and oracle rows. The mixed
and uncapped-only 1F1B big profiles are opt-in and are not included in `all`.

## Seeds

`uniform-breadth-first` and `uniform-interleaved-1f1b` are deterministic and
appear once per configuration and activation mode. Solver-backed methods use
seeds `0,1,2` by default outside the smoke/oracle profiles. Every row still
records `random_seed` explicitly.

## Equal Memory

Equal-memory rows use:

- `activation_model = linear_in_stage_layers`
- `activation_units_per_layer = 1`
- `activation_cap_mode = uniform_baseline`
- `enforce_activation_cap = true`

The generator materializes one W-length uniform-baseline cap vector per
configuration. It uses the deterministic uniform partition and breadth-first
worker order, not CP-SAT or 1F1B, and records `activation_cap_baseline_method`,
the cap vector, and the derivation hash in every row for that configuration. The
runner passes those fields with
`--activation-cap-mode uniform-baseline`, `--activation-cap-units`, and
`--activation-cap-derivation-hash`.

For deterministic fixed-order methods, equal-memory enforcement is a
deterministic postconstruction check. Solver-backed equal-memory rows must
record solver enforcement and added activation constraints; post-hoc-only rows
are rejected by the compatibility checker.

## Runner

Generate and run smoke:

```bash
python3 scripts/generate_cal_manifest.py --profile smoke \
  --output manifests/cal_smoke.jsonl \
  --binary build/ortools/slackpipe_cli

python3 scripts/run_cal_manifest.py \
  --manifest manifests/cal_smoke.jsonl \
  --output-root results/cal_smoke \
  --binary build/ortools/slackpipe_cli \
  --jobs 1
```

The runner supports `--group`, `--method`, `--configuration-id`, `--run-id`,
`--jobs`, `--dry-run`, `--force`, `--fail-fast`, and
`--no-rerun-invalid`.

The runner writes `.slackpipe_cal_manifest.json` in each non-dry-run result
root. If that root already contains a different manifest hash, the runner
refuses to append unless `--force-new-manifest` is supplied. This prevents
accidental mixing of results from different protocol snapshots.

Completed rows are skipped only after the shared checker verifies schema
versions, method contract hash, manifest hash/run id, git policy, controlled
problem fields, activation fields, cap eligibility, validation status, seed,
threads, and time limit. Invalid/malformed/mismatched rows are rerun unless
`--no-rerun-invalid` is supplied.

Outcome classes:

- `completed_valid`
- `completed_invalid`
- `unavailable`
- `timeout_or_failed_process`
- `skipped_valid`
- `rerun_invalid`
- `schema_mismatch`
- `manifest_mismatch`

`UNAVAILABLE` is never counted as a successful result. OR-Tools-disabled builds
write solver-backed rows as unavailable rather than running uncapped substitutes.

## Analysis

After rows have been produced by the runner, analyze the same manifest/result
root pair with the deterministic CAL analyzer:

```bash
python3 scripts/analyze_cal_results.py \
  --manifest manifests/cal_smoke.jsonl \
  --results-root results/cal_smoke \
  --output-dir results/cal_smoke_analysis \
  --mode both \
  --include-groups smoke \
  --strict
```

The analyzer reuses the same compatibility checker as the runner, writes
preflight row classifications, normalizes makespan to the
`uniform-breadth-first` row for each workload, aggregates seeds by median, and
serializes all derived values as CSV and JSON. See
[`evaluation_analysis.md`](evaluation_analysis.md) for the figure, table,
oracle, convergence, and equal-memory eligibility rules.

Production command, not run by default:

```bash
python3 scripts/generate_cal_manifest.py --profile main \
  --output manifests/cal_main.jsonl \
  --binary build/ortools/slackpipe_cli \
  --time-limit-seconds 300 \
  --solver-threads 1

python3 scripts/run_cal_manifest.py \
  --manifest manifests/cal_main.jsonl \
  --output-root results/cal_main \
  --binary build/ortools/slackpipe_cli \
  --jobs 1
```

Opt-in 1F1B production command, not run by default:

```bash
python3 scripts/generate_cal_manifest.py --profile main_big_5min_1f1b \
  --output manifests/cal_main_big_5min_1f1b.jsonl \
  --binary build/ortools/slackpipe_cli

python3 scripts/run_cal_manifest.py \
  --manifest manifests/cal_main_big_5min_1f1b.jsonl \
  --output-root results/cal_main_big_5min_1f1b \
  --binary build/ortools/slackpipe_cli \
  --jobs 1
```

Opt-in uncapped-only 1F1B production command, not run by default:

```bash
python3 scripts/generate_cal_manifest.py --profile main_big_5min_1f1b_uncapped \
  --output manifests/cal_main_big_5min_1f1b_uncapped.jsonl \
  --binary build/ortools/slackpipe_cli

python3 scripts/run_cal_manifest.py \
  --manifest manifests/cal_main_big_5min_1f1b_uncapped.jsonl \
  --output-root results/cal_main_big_5min_1f1b_uncapped \
  --binary build/ortools/slackpipe_cli \
  --jobs 1
```
