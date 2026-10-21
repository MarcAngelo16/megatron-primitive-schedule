# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in compute-stream timing evidence, never a solver cost estimator."""

import time
from contextlib import contextmanager, nullcontext

import torch

from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator


class ComputeTimingDiagnostic:
    """Reuse preinitialized event pairs; resolve only at full-step completion.

    An event interval is a stream envelope, not summed kernel time. It can
    include host launch gaps and dependencies inserted inside the region. Work
    on other streams is covered only if it rejoins the recorded compute stream.
    """

    def __init__(self, operation_capacity: int, *, serialized=False, timeline=False, nvtx=False):
        if operation_capacity < 1:
            raise ValueError("Positive operation capacity required")
        self.serialized = serialized
        self.timeline = timeline
        self.nvtx = nvtx
        self.step_pair = (
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            if timeline
            else None
        )
        self.pairs = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(operation_capacity)
        ]
        # CUDA events are lazily initialized. Pay that cost before training warmup.
        for pair in self.pairs + ([self.step_pair] if timeline else []):
            for event in pair:
                event.record()
        torch.cuda.synchronize()
        self.active = False
        self.iterations = []

    @contextmanager
    def capture(self, path):
        """One raw capture per partition, including training warmup; never fitted.

        Export only after all measured iterations. No trace serialization between
        samples. Profiling is intentionally optional because CUPTI perturbs work.
        """
        if not self.timeline:
            yield
            return
        if path.exists():
            raise ValueError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        ) as profiler:
            yield
        profiler.export_chrome_trace(str(path))

    @contextmanager
    def region(self, name):
        with (
            torch.profiler.record_function(name) if self.timeline else nullcontext(),
            torch.cuda.nvtx.range(name) if self.nvtx else nullcontext(),
        ):
            yield

    def begin(self, iteration: int) -> None:
        if self.active:
            raise RuntimeError("Timing iteration already active")
        self.active = True
        self.iteration = iteration
        self.records = []
        self.p2p = []
        self.position = 0
        if self.step_pair:
            self.step_pair[0].record()

    def compute(self, compute, identity: dict):
        if not self.active or len(self.records) >= len(self.pairs):
            raise RuntimeError("Inactive diagnostic or exhausted preallocated event capacity")
        start, end = self.pairs[len(self.records)]
        stream = torch.cuda.current_stream()
        position = self.position
        self.position += 1
        drain_ms = None
        if self.serialized:
            before = time.perf_counter()
            torch.cuda.synchronize()
            drain_ms = (time.perf_counter() - before) * 1000
        kind = "F" if identity["phase"] == "forward_compute" else "B"
        label = f"ScheduleTrace/{kind}/i{self.iteration}/s{identity.get('logical_stage', 0)}/b{identity.get('microbatch', 0)}"
        with self.region(label):
            start.record(stream)
            before = time.perf_counter()
            result = compute()
            cpu_ms = (time.perf_counter() - before) * 1000
            end.record(stream)
            if self.serialized:
                torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - before) * 1000
        self.records.append(
            dict(
                **identity,
                iteration=self.iteration,
                schedule_position=position,
                compute_position=len(self.records),
                compute_stream_id=stream.cuda_stream,
                cpu_wall_ms=cpu_ms,
                wall_ms=wall_ms,
                wall_definition=(
                    "host closure plus completion"
                    if self.serialized
                    else "host closure (asynchronous)"
                ),
                diagnostic_execution="serialized" if self.serialized else "normal",
                pre_compute_drain_ms=drain_ms,
                existing_stage_wall_ms=None,
                wait_synchronization_ms=None,
            )
        )
        if self.nvtx:
            self.records[-1].update(
                start_event_handle=start.cuda_event, end_event_handle=end.cuda_event
            )
        return result

    def finish(self) -> list:
        """One device completion boundary, after optimizer; no per-op waits."""
        if not self.active:
            raise RuntimeError("No timing iteration active")
        try:
            if self.step_pair:
                self.step_pair[1].record()
            before = time.perf_counter()
            torch.cuda.synchronize()
            completion_ms = (time.perf_counter() - before) * 1000
            for record, (start, end) in zip(self.records, self.pairs):
                record["compute_gpu_ms"] = start.elapsed_time(end)
            self.iterations.append(
                dict(
                    iteration=self.iteration,
                    completion_wait_ms=completion_ms,
                    p2p_wall_ms=sum(e["p2p_wall_ms"] for e in self.p2p),
                    p2p_calls=self.p2p,
                    compute_operation_count=len(self.records),
                    cuda_elapsed_ms=(
                        self.step_pair[0].elapsed_time(self.step_pair[1])
                        if self.step_pair
                        else None
                    ),
                )
            )
            return self.records
        finally:
            self.active = False


class DiagnosticP2PCommunicator(P2PCommunicator):
    """Observe the native call boundary without changing its requests or waits."""

    def __init__(self, pp_group, config, diagnostic: ComputeTimingDiagnostic):
        if diagnostic.serialized and (
            not config.batch_p2p_comm or not config.batch_p2p_sync or config.overlap_p2p_comm
        ):
            raise ValueError(
                "Serialized diagnostic requires blocking batch P2P with native batch sync and no pending async requests"
            )
        super().__init__(pp_group, config)
        self.diagnostic = diagnostic

    def _communicate(self, **kwargs):
        diagnostic = self.diagnostic
        if not diagnostic.active:
            return super()._communicate(**kwargs)
        position = diagnostic.position
        diagnostic.position += 1
        serialized = getattr(diagnostic, "serialized", False)
        if serialized and not kwargs.get("wait_on_reqs", True):
            raise ValueError("Serialized diagnostic cannot drain unmatched asynchronous P2P")
        before = time.perf_counter()
        region = (
            diagnostic.region(f"SlackPipeDiagnostic/P2P/i{diagnostic.iteration}/p{position}")
            if getattr(diagnostic, "timeline", False) or getattr(diagnostic, "nvtx", False)
            else nullcontext()
        )
        with region:
            result = super()._communicate(**kwargs)
            if serialized:
                torch.cuda.synchronize()
        elapsed = (time.perf_counter() - before) * 1000
        diagnostic.p2p.append(
            dict(
                schedule_position=position,
                p2p_wall_ms=elapsed,
                # This inclusive host duration cannot be decomposed into NCCL
                # device service and dependency wait without a separate trace.
                wait_synchronization_ms=None,
                recv_prev=kwargs["recv_prev"],
                recv_next=kwargs["recv_next"],
                send_prev=kwargs["tensor_send_prev"] is not None,
                send_next=kwargs["tensor_send_next"] is not None,
                wait_on_reqs=kwargs.get("wait_on_reqs", True),
                native_batch_sync=bool(self.config.batch_p2p_comm and self.config.batch_p2p_sync),
            )
        )
        return result


def add_overlap_diagnostic_arguments(parser):
    parser.add_argument(
        "--diagnostic-nvtx",
        action="store_true",
        help="Requires --diagnose-compute-timing; operation/P2P NVTX labels and event handles for external Nsight capture",
    )
    parser.add_argument(
        "--diagnostic-execution",
        choices=("normal", "serialized"),
        default="normal",
        help="Requires --diagnose-compute-timing; serialized drains the device around compute for diagnosis only",
    )
    parser.add_argument(
        "--diagnostic-timeline",
        action="store_true",
        help="Requires --diagnose-compute-timing; capture CPU/CUDA activity for every warmup and measured operation",
    )
