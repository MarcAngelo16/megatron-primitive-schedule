# Current Five-method Evaluation Protocol

The authoritative current [method/workflow guide](../docs/slackpipe_five_methods.md)
defines five methods: native `1f1b`, native `interleaved-1f1b`,
`octopipe-tuning`, `cpsat-no-hint`, `slack-heuristic`.

Fix model, precision, data, B, physical workers, transport and accepted cost
profile. Fix explicit N and cyclic placement for the three plan generators and
native interleaving. Native noninterleaved 1F1B uses N=W by definition.
Use one search worker and the same wall-clock budget; multi-worker CP-SAT is a
separate explicit experiment. Record initialization/search/validation scope and
actual elapsed time, not only the configured budget.

No external hints/incumbent/fallback or search-space restrictions in CP-SAT.
No placement tuning or new phase fallback in the restricted OctoPipe comparison.
No heuristic-to-CP-SAT combination. Enforced activation caps are CP-SAT-only;
post-hoc memory analysis is not enforcement.

Require accepted calibration quality, independent schedule replay and validator,
plan provenance, and GPU smoke before benchmark/trace. Preserve no-solution,
unsupported, timeout and quality-failure outcomes. Do not generate fake plans.
Use immutable attempt directories, v3 context-bound receipts and content hashes.
Historical hinted results cannot be resumed as unhinted results.

Benchmark with the profiler disabled; collect diagnostic timelines separately.
Report independent-run variability, all raw samples, continuous synchronized
step wall time and measured peaks. Predicted F/B makespan, GPU step time and
solver wall time have different boundaries. Do not label simulated improvements
as measured speedups or reinterpret FEASIBLE as OPTIMAL.

The default staged driver retains its existing configurable warmup/measurement
policy; formal throughput runs should explicitly request 30 warmups, 50
measurements and three independent runs, checking stability before comparison.
No performance sweep is part of the refactor validation.

The former CAL/ablation protocol is historical. Use the original result's
recorded commit and configuration to reproduce it; raw data and receipts are
not rewritten by this simplification.
