# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Receipt-gated, immutable real-system experiments; run inside the project container."""

import json
import os
import sys
import uuid
from pathlib import Path

from tools.slackpipe_eval_args import argument_parser, normalize_args
from tools.slackpipe_eval_common import (
    DETERMINISTIC,
    ROOT,
    digest,
    require_cuda_devices,
    run,
    source_identity,
    write_json,
)
from tools.slackpipe_eval_config import (
    METHODS,
    PLAN_SCHEDULES,
    fingerprint,
    load_model,
    parameter_breakdown,
    schedule_topology,
    solver_arguments,
)
from tools.slackpipe_eval_receipts import (
    RECEIPT_SCHEMA,
    archive_receipt,
    assess_receipt,
    parent_identity,
)
from tools.slackpipe_profiling import (
    collect_profiles,
    profiling_policy,
    stage_calibration_schedule,
    validate_collection_capabilities,
)

STAGES = (
    "env",
    "correctness",
    "smoke",
    "calibrate",
    "solve",
    "benchmark",
    "trace",
    "full",
    "inspect",
)


def memory_preflight(
    model: dict, topology: dict, precision: str, seq: int, micro: int, devices: list
) -> dict:
    counts = parameter_breakdown(model)
    pp, stages, layers = topology["pp"], topology["num_stages"], model["num_layers"]
    cuts = [layers * s // stages for s in range(stages + 1)]
    per_rank = [0] * pp
    for s, (a, b) in enumerate(zip(cuts, cuts[1:])):
        per_rank[s % pp] += sum(counts["layers"][a:b])
    per_rank[0] += counts["embedding"]
    per_rank[(stages - 1) % pp] += counts["output"] + counts["final_norm"]
    element = 2 if precision == "bf16" else 4
    parameter_gradient_bytes = [p * element * 2 for p in per_rank]
    # Saved tensors vary by kernel, partition and live microbatches: this is not a fit guarantee.
    activation_estimate = (
        seq
        * micro
        * model["hidden_size"]
        * element
        * 32
        * max(b - a for a, b in zip(cuts, cuts[1:]))
        * topology["num_microbatches"]
    )
    status = "passed"
    if len(devices) < pp:
        status = "skipped_hardware_capacity"
    elif any(
        need > device["free_bytes"] * 0.9 for need, device in zip(parameter_gradient_bytes, devices)
    ):
        status = "skipped_memory_capacity"
    return dict(
        status=status,
        parameter_gradient_bytes_by_rank=parameter_gradient_bytes,
        activation_estimate_bytes_per_rank=activation_estimate,
        devices=devices,
        policy="Skip if parameters+gradients exceed 90% free memory; activation estimate is advisory, not a capacity certificate. No automatic model/sequence shrink.",
    )


def benchmark_summary(results_by_run: list, global_batch_size: int, seq_length: int) -> dict:
    from megatron.core.pipeline_parallel.slackpipe.collection import summarize_rank_samples
    from tools.slackpipe_eval_common import summarize_samples

    runs = [summarize_rank_samples(results) for results in results_by_run]
    event_summary = summarize_samples([r["max_rank_iteration_ms"] for r in runs])
    walls = [r["continuous_wall_ms"] / len(r["iterations"]) for r in runs]
    summary = summarize_samples([[wall] for wall in walls])
    if len(runs) == 1:
        summary["ci95_ms"] = None
        summary["rep_mean_stddev_ms"] = None
        event_summary["ci95_ms"] = None
        event_summary["rep_mean_stddev_ms"] = None
    wall_mean = sum(walls) / len(walls)
    memories = [
        m for results in results_by_run for r in results for m in r["memory_boundaries"].values()
    ]
    return dict(
        schema_version="slackpipe.eval_benchmark.v2",
        **summary,
        summary_metric="max-rank continuous window wall time / measured steps",
        cuda_event_summary=event_summary,
        config=results_by_run[0][0]["config"],
        collection_provenance=[
            dict(
                environment=r[0].get("environment"),
                options=r[0]["options"],
                timing_definitions=r[0].get("timing_definitions"),
            )
            for r in results_by_run
        ],
        runs=runs,
        timed_steps_by_run=[len(r["iterations"]) for r in runs],
        continuous_window_observations=len(walls),
        continuous_step_ms_by_run=walls,
        continuous_mean_ms=wall_mean,
        continuous_ci95_ms=summary["ci95_ms"],
        samples_per_second=global_batch_size * 1000 / wall_mean,
        tokens_per_second=global_batch_size * seq_length * 1000 / wall_mean,
        peak_allocated_bytes=max(m["peak_allocated_bytes"] for m in memories),
        peak_reserved_bytes=max(m["peak_reserved_bytes"] for m in memories),
        ci_definition="95% normal-approximation half-width across independent continuous-window step means; null for one run",
        throughput_definition="synchronized continuous wall window; not per-operation trace envelopes",
    )


class Experiment:
    """Stage receipts bind source, model, measurement policy and prerequisite bytes."""

    def __init__(self, args, *, shared_receipts=()):
        self.args = normalize_args(args)
        self.model = load_model(args.model_config)
        self.output = args.output
        self.topology = schedule_topology(
            args.schedule, args.pp, args.logical_stages, args.microbatches
        )
        self.env = dict(os.environ, **DETERMINISTIC)
        self.env.pop("SLACKPIPE_HYBRID_PLAN", None)
        self.env["PYTHONPATH"] = str(ROOT) + os.pathsep + self.env.get("PYTHONPATH", "")
        self.torchrun = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.pp}",
        ]
        self.completed = {}
        self._identity = None
        # Only receipts accepted earlier in this campaign may be shared without --resume.
        self.shared_receipts = frozenset(Path(p).resolve() for p in shared_receipts)

    def _base_context(self):
        from megatron.core.pipeline_parallel.slackpipe.collection import environment_metadata

        if self._identity is not None:
            return self._identity
        args = self.args
        import torch

        available = torch.cuda.is_available()
        environment = environment_metadata() if available else dict(cuda_available=False)
        environment["visible_gpu_uuids"] = [
            str(torch.cuda.get_device_properties(i).uuid)
            for i in range(torch.cuda.device_count() if available else 0)
        ]
        environment["worker_deterministic_environment"] = DETERMINISTIC
        self._identity = dict(
            model_config_hash=fingerprint(self.model),
            topology=self.topology,
            precision=args.precision,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            transport=args.transport,
            source=source_identity(),
            environment=environment,
        )
        return self._identity

    def _context(self, stage, parents):
        """Fingerprint actual stage inputs, never a campaign-wide measurement policy."""
        args = self.args
        context = dict(self._base_context(), context_schema_version=4, stage=stage)
        context["parents"] = {k: parent_identity(v) for k, v in parents.items()}
        if stage not in ("env", "correctness"):
            context.update(seed=args.seed, learning_rate=args.learning_rate)
        if stage not in ("env", "correctness", "solve", "calibrate"):
            context["schedule"] = (
                ("1f1b" if self.topology["num_stages"] == args.pp else "interleaved-1f1b")
                if stage == "native-smoke"
                else args.schedule
            )
        policy = {}
        if stage in ("smoke", "native-smoke"):
            policy = dict(warmups=args.smoke_warmups, iterations=args.smoke_iterations)
        elif stage == "calibrate":
            policy = profiling_policy(args)
            context["calibration_schedules"] = {
                backend: identity.get("calibration_schedule", "artifact-import")
                for backend, identity in policy["backends"].items()
            }
            policy["stage_partitions"] = (
                "class-identifiable-v1" if self.model["model_family"] == "nemotron_h" else "uniform"
            )
        elif stage == "solve":
            context.update(
                solver_sha256=digest(args.solver) if args.solver.is_file() else None,
                solver_seconds=args.solver_seconds,
                method_contract=METHODS[args.schedule].contract(),
                solver_algorithm=METHODS[args.schedule].backend,
                solver_workers=args.solver_workers,
                activation_cap_units=args.activation_cap_units,
                require_optimal=False,
                ratio=[1, 1],
                communication_model="constant_inter_worker_dependency_delay",
                communication_ticks=0,
            )
            if args.schedule == "octopipe-tuning":
                context.update(
                    octopipe_fixed_placement=True, octopipe_initial_order="uniform-breadth-first"
                )
        elif stage == "benchmark":
            policy = dict(warmups=args.warmups, iterations=args.iterations)
            context.update(
                run_index=args.run_index,
                repetitions=args.repetitions if args.run_index is None else 1,
            )
        elif stage == "trace":
            policy = dict(
                warmups=args.trace_warmups,
                **{
                    k: getattr(args, k)
                    for k in (
                        "profiler_wait",
                        "profiler_warmup",
                        "profiler_active",
                        "profiler_repeat",
                    )
                },
                trace_format="slackpipe.figure_trace.v2",
                profiler="cpu+cuda;no-shapes-stack-memory",
            )
        if stage in ("solve", "smoke", "benchmark", "trace"):
            context["method_contract"] = METHODS[args.schedule].contract()
            context["executor"] = METHODS[args.schedule].executor
        context["policy"] = policy
        return context

    def dependencies(self, stage):
        optimized = self.args.schedule in PLAN_SCHEDULES
        return {
            "env": (),
            "correctness": ("env",),
            "native-smoke": ("correctness",),
            "calibrate": ("correctness",),
            "solve": ("calibrate",),
            "smoke": ("solve",) if optimized else ("correctness",),
            "benchmark": ("smoke",),
            "trace": ("smoke",),
        }[stage]

    def receipt_path(self, stage):
        index = self.args.run_index
        suffix = f".run{index:03d}" if stage == "benchmark" and index is not None else ""
        return self.output / "receipts" / f"{self.args.schedule}.{stage}{suffix}.json"

    def _execute(self, command, directory, name):
        run(command, directory / f"{name}.log", self.args.timeout, self.env)

    def _worker(
        self,
        stage,
        directory,
        *,
        schedule=None,
        plan=None,
        profile=None,
        iterations=None,
        backend="stage",
    ):
        args = self.args
        if stage == "calibrate" and backend == "stage":
            native = stage_calibration_schedule(args)
            if schedule is not None and schedule != native:
                raise ValueError("Stage calibration must use native interleaved-1f1b")
            schedule = native
        warmups = {
            "calibrate": args.calibration_warmups,
            "smoke": args.smoke_warmups,
            "trace": args.trace_warmups,
        }.get(stage, args.warmups)
        profiler = (
            (args.profiler_wait, args.profiler_warmup, args.profiler_active, args.profiler_repeat)
            if stage == "trace"
            else (2, 2, 3, 3)
        )
        measured = (
            sum(profiler[:3]) * profiler[3] if stage == "trace" else (iterations or args.iterations)
        )
        command = self.torchrun + [
            "-m",
            "tools.slackpipe_eval_worker",
            "--phase",
            stage,
            "--model-config",
            args.model_config,
            "--method",
            schedule or args.schedule,
            "--pp",
            args.pp,
            "--num-stages",
            self.topology["num_stages"],
            "--microbatches",
            args.microbatches,
            "--micro-batch-size",
            args.micro_batch_size,
            "--seq-length",
            args.seq_length,
            "--precision",
            args.precision,
            "--transport",
            args.transport,
            "--seed",
            args.seed,
            "--learning-rate",
            args.learning_rate,
            "--warmups",
            warmups,
            "--iterations",
            measured,
            "--output",
            directory,
            "--run-id",
            f"{directory.parent.name}-{directory.name}",
            "--profiler-wait",
            profiler[0],
            "--profiler-warmup",
            profiler[1],
            "--profiler-active",
            profiler[2],
            "--profiler-repeat",
            profiler[3],
        ]
        if plan:
            command += ["--plan", plan, "--profile", profile]
        if stage == "calibrate":
            command += ["--profiling-method", backend]
            if backend == "layer":
                command = [
                    sys.executable,
                    "-m",
                    "tools.slackpipe_isolated_profile",
                    *command[command.index("tools.slackpipe_eval_worker") + 1 :],
                ]
                for name in ("isolated_profile_warmups", "isolated_profile_iterations"):
                    command += ["--" + name.replace("_", "-"), getattr(args, name)]
            else:
                command += [
                    "--stage-residual-k",
                    args.stage_residual_k,
                    "--stage-residual-rho",
                    args.stage_residual_rho,
                ]
        self._execute(command, directory.parent, directory.name)

    def ensure(self, stage):
        validate_collection_capabilities(self.args, stage)
        if stage == "env":
            # This launcher uses single-node torchrun with TP=DP=CP=1.
            # Check before receipt reuse/context queries or starting any workers.
            require_cuda_devices(self.args.pp)
        if stage in self.completed:
            return self.completed[stage]
        args = self.args
        optimized = args.schedule in PLAN_SCHEDULES
        if stage == "solve" and not optimized:
            return dict(
                status="not_applicable",
                stage=stage,
                method_id=args.schedule,
                executor=METHODS[args.schedule].executor,
                solver=None,
            )
        parents = {p: self.ensure(p) for p in self.dependencies(stage)}
        if any(p["status"] != "passed" for p in parents.values()):
            result = dict(
                schema_version=RECEIPT_SCHEMA,
                status="skipped_prerequisite",
                stage=stage,
                parents=parents,
            )
            path = self.receipt_path(stage)
            if path.exists():
                archive_receipt(path, json.loads(path.read_text()))
            write_json(path, result)
            self.completed[stage] = result
            return result
        context = self._context(stage, parents)
        receipt_path = self.receipt_path(stage)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            reusable = args.resume or (args.force and stage != args.stage and args.stage != "full")
            state, reason, accepted = assess_receipt(receipt, context, self.output, parents)
            if args.explain_receipt:
                print(f"{receipt_path}: {state}: {reason}")
            if (
                reusable
                and not (args.force and args.stage in (stage, "full"))
                and state in ("compatible", "legacy-compatible")
            ):
                if state == "legacy-compatible":
                    archive_receipt(receipt_path, receipt)
                    write_json(receipt_path, accepted)
                    print(f"Migrated {receipt_path}: {reason}; artifact bytes unchanged")
                self.completed[stage] = accepted
                return accepted
            if not args.force and not (args.resume and state == "stale"):
                raise RuntimeError(
                    f"Existing/stale receipt: {receipt_path}: {state}: {reason}; "
                    "use --resume for compatible reuse/new v2 context, or --force for an explicit new attempt"
                )
            archive_receipt(receipt_path, receipt)
        elif args.resume or self.shared_receipts:
            # Shared native prerequisites keep separate schedule namespaces, but reuse exact bytes.
            if stage in ("env", "correctness", "native-smoke", "calibrate", "solve"):
                for candidate in sorted(receipt_path.parent.glob(f"*.{stage}.json")):
                    if not args.resume and candidate.resolve() not in self.shared_receipts:
                        continue
                    receipt = json.loads(candidate.read_text())
                    state, _, _ = assess_receipt(receipt, context, self.output, parents)
                    if state == "compatible":
                        receipt = dict(
                            receipt, schedule=args.schedule, reused_from=str(candidate.name)
                        )
                        write_json(receipt_path, receipt)
                        self.completed[stage] = receipt
                        return receipt
        folder = {"calibrate": "calibration", "solve": "solver", "trace": "traces"}.get(
            stage, args.schedule
        )
        directory = self.output / folder / f"{stage}-{uuid.uuid4().hex[:12]}"
        directory.mkdir(parents=True)
        config_path = self.output / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != self.model:
            raise ValueError("Output belongs to a different model; use a different directory")
        write_json(config_path, self.model)
        result = dict(
            schema_version=RECEIPT_SCHEMA,
            status="passed",
            stage=stage,
            context=context,
            context_hash=fingerprint(context),
            schedule=args.schedule,
            method_id=args.schedule,
            executor=METHODS[args.schedule].executor,
            transport=args.transport if optimized else "megatron-p2p",
            solver_profile=args.solver_profile if optimized else None,
            directory=str(directory.relative_to(self.output)),
        )
        if not optimized:
            result["solver"] = None
        if optimized and stage in ("solve", "smoke", "benchmark", "trace"):
            result["profile_identity"] = self.completed["calibrate"]["profile_identity"]
        plan = profile = None
        if optimized and stage in ("smoke", "benchmark", "trace"):
            solver = self.completed["solve"]
            plan = self.output / solver["plan"]
            profile = self.output / self.completed["calibrate"]["profile"]
        try:
            if stage == "env":
                import torch

                devices = []
                for i in range(min(torch.cuda.device_count(), args.pp)):
                    free, total = torch.cuda.mem_get_info(i)
                    prop = torch.cuda.get_device_properties(i)
                    devices.append(
                        dict(
                            name=prop.name, uuid=str(prop.uuid), free_bytes=free, total_bytes=total
                        )
                    )
                preflight = memory_preflight(
                    self.model,
                    self.topology,
                    args.precision,
                    args.seq_length,
                    args.micro_batch_size,
                    devices,
                )
                result["status"] = preflight["status"]
                write_json(directory / "memory_preflight.json", preflight)
            elif stage == "correctness":
                self.env["SLACKPIPE_TEST_TRANSPORT"] = args.transport
                # Small architecture fixtures validate this world/precision, not full-size convergence.
                target = f"tests/unit_tests/pipeline_parallel/test_slackpipe_hybrid.py::test_hybrid_numerical_equivalence[{args.precision}-{args.pp}]"
                self._execute(
                    self.torchrun
                    + ["-m", "pytest", target, "-q", "--junitxml", directory / "correctness.xml"],
                    directory,
                    "correctness",
                )
                import xml.etree.ElementTree as ET

                tree = ET.parse(directory / "correctness.xml")
                cases = tree.findall(".//testcase")
                if not cases or any(c.find("skipped") is not None for c in cases):
                    raise RuntimeError("Correctness gate skipped; no acceptance")
                result["scope"] = (
                    "small hybrid numerical-equivalence fixture at requested PP/precision; target model validated separately by smoke"
                )
            elif stage == "calibrate":
                from tools.slackpipe_eval_worker import validate_profile

                collection = collect_profiles(
                    directory,
                    args,
                    lambda backend, worker: self._worker(
                        "calibrate",
                        worker,
                        schedule="interleaved-1f1b" if backend == "stage" else args.schedule,
                        iterations=args.calibration_iterations,
                        backend=backend,
                    ),
                    lambda profile: validate_profile(
                        profile,
                        self.model,
                        self.topology,
                        args.precision,
                        args.seq_length,
                        args.micro_batch_size,
                    ),
                )
                result["collection_status"] = collection["status"]
                result["profiling_method"] = args.profiling_method
                result["solver_profile"] = args.solver_profile
                result["backend_status"] = {
                    key: value["status"] for key, value in collection["backends"].items()
                }
                result["profile"] = str(
                    Path(collection["selected_profile"]).relative_to(self.output)
                )
                selected_backend = collection["backends"][args.solver_profile]
                result["profile_identity"] = {
                    key: selected_backend[key]
                    for key in ("backend", "sha256", "cost_profile_hash", "policy")
                }
            elif stage in ("smoke", "native-smoke"):
                native = "1f1b" if self.topology["num_stages"] == args.pp else "interleaved-1f1b"
                self._worker(
                    "smoke" if stage == "native-smoke" else stage,
                    directory / "worker",
                    schedule=native if stage == "native-smoke" else args.schedule,
                    plan=plan,
                    profile=profile,
                    iterations=args.smoke_iterations,
                )
            elif stage == "solve":
                from megatron.core.pipeline_parallel.slackpipe.plan import (
                    load_slackpipe_plan,
                    validate_cyclic_placement,
                )
                from tools.slackpipe_eval_worker import validate_profile

                profile = self.output / self.completed["calibrate"]["profile"]
                profile_data = json.loads(profile.read_text())
                validate_profile(
                    profile_data,
                    self.model,
                    self.topology,
                    args.precision,
                    args.seq_length,
                    args.micro_batch_size,
                )
                plan = directory / "slackpipe.plan.json"
                self._execute(
                    [
                        args.solver,
                        *solver_arguments(
                            args.schedule,
                            args.solver_seconds,
                            args.solver_workers,
                            args.activation_cap_units,
                        ),
                        "--B",
                        args.microbatches,
                        "--N",
                        self.topology["num_stages"],
                        "--J",
                        args.pp,
                        "--L",
                        self.model["num_layers"],
                        "--ratio-num",
                        1,
                        "--ratio-den",
                        1,
                        "--communication",
                        0,
                        "--seed",
                        args.seed,
                        "--cost-profile",
                        profile,
                        "--output-prefix",
                        directory / "solver",
                        "--emit-plan",
                        plan,
                    ],
                    directory,
                    "solve",
                )
                outcome = json.loads((directory / "solver.json").read_text())
                if outcome["method_id"] != args.schedule or outcome["schema_version"] != 2:
                    raise ValueError("Solver method contract mismatch")
                result["solver_result"] = str((directory / "solver.json").relative_to(self.output))
                if not outcome["final_solution_available"]:
                    if plan.exists():
                        raise ValueError("No-solution solver unexpectedly emitted a plan")
                    result.update(
                        status="no_valid_solution", solver_status_raw=outcome["solver_status_raw"]
                    )
                else:
                    self._execute(
                        [
                            args.solver,
                            "--validate-result",
                            directory / "solver.json",
                            "--validation-output",
                            directory / "independent_validation.json",
                        ],
                        directory,
                        "validate",
                    )
                    parsed = load_slackpipe_plan(plan, pipeline_model_parallel_size=args.pp)
                    validate_cyclic_placement(parsed)
                    if (parsed.num_layers, parsed.num_stages, parsed.num_microbatches) != (
                        self.model["num_layers"],
                        self.topology["num_stages"],
                        args.microbatches,
                    ):
                        raise ValueError("Solver plan dimensions differ from experiment")
                    if (
                        parsed.schema_version == "slackpipe.plan.v2"
                        and parsed.cost_profile_hash != profile_data["cost_profile_hash"]
                    ):
                        raise ValueError("Solver plan profile provenance mismatch")
                    if parsed.schema_version == "slackpipe.plan.v1":
                        write_json(
                            plan.with_suffix(".binding.json"),
                            dict(
                                schema_version="slackpipe.eval_plan_binding.v1",
                                plan_sha256=digest(plan),
                                profile_sha256=digest(profile),
                            ),
                        )
                    from tools.slackpipe_eval_worker import validate_plan_provenance

                    validate_plan_provenance(plan, profile, profile_data)
                    result["plan"] = str(plan.relative_to(self.output))
            else:
                runs = []
                repetitions = (
                    args.repetitions if stage == "benchmark" and args.run_index is None else 1
                )
                for rep in range(repetitions):
                    worker = directory / f"run{rep:03d}"
                    self._worker(stage, worker, plan=plan, profile=profile)
                    runs.append(
                        [
                            json.loads((worker / f"result.rank{r}.json").read_text())
                            for r in range(args.pp)
                        ]
                    )
                if stage == "benchmark":
                    summary = benchmark_summary(
                        runs, args.microbatches * args.micro_batch_size, args.seq_length
                    )
                    write_json(directory / "summary.json", summary)
                    result["summary"] = str((directory / "summary.json").relative_to(self.output))
            write_json(
                directory / "stage.json", {k: v for k, v in result.items() if k != "context"}
            )
            result["artifacts"] = {
                str(p.relative_to(self.output)): digest(p)
                for p in directory.rglob("*")
                if p.is_file()
            }
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            result["artifacts"] = {
                str(p.relative_to(self.output)): digest(p)
                for p in directory.rglob("*")
                if p.is_file()
            }
            write_json(receipt_path, result)
            raise
        write_json(receipt_path, result)
        self.completed[stage] = result
        return result

    def execute(self):
        if self.args.stage == "inspect":
            return self.inspect()
        if self.args.stage == "full":
            benchmark = self.ensure("benchmark")
            if benchmark["status"] != "passed":
                return benchmark
        return self.ensure("trace" if self.args.stage == "full" else self.args.stage)

    def inspect(self):
        """Read-only recursive compatibility audit, without launching or rewriting stages."""
        reports, accepted = {}, {}

        def visit(stage):
            if stage in reports:
                return
            for parent in self.dependencies(stage):
                visit(parent)
            path = self.receipt_path(stage)
            row = dict(stage=stage, receipt=str(path), status="missing", compatible=False)
            reports[stage] = row
            if not path.exists():
                row["reason"] = "receipt missing"
                return
            receipt = json.loads(path.read_text())
            row.update(
                schema=receipt.get("schema_version"),
                context_hash=fingerprint(receipt.get("context")),
                artifacts=receipt.get("artifacts"),
                status=receipt.get("status"),
            )
            if any(p not in accepted for p in self.dependencies(stage)):
                row["reason"] = "incompatible/missing prerequisite"
                return
            parents = {p: accepted[p] for p in self.dependencies(stage)}
            state, reason, candidate = assess_receipt(
                receipt, self._context(stage, parents), self.output, parents
            )
            row.update(
                compatibility=state,
                reason=reason,
                compatible=state in ("compatible", "legacy-compatible"),
            )
            if row["compatible"]:
                accepted[stage] = candidate

        stages = (
            ("solve", "benchmark", "trace")
            if self.args.schedule in PLAN_SCHEDULES
            else ("benchmark", "trace")
        )
        for stage in stages:
            visit(stage)
        return dict(status="passed", inspection=list(reports.values()), read_only=True)


def main() -> None:
    result = Experiment(argument_parser().parse_args()).execute()
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
