"""GPU prerequisite branches only; mocked devices are not hardware acceptance."""

import os
import subprocess
import sys

import pytest
import torch

from tests.unit_tests.test_utilities import Utils
from tools.run_slackpipe_eval import Experiment
from tools.slackpipe_eval_args import argument_parser, normalize_args
from tools.slackpipe_eval_common import ROOT, require_cuda_devices


def args_for(tmp_path, *, phase="benchmark", method="interleaved-1f1b"):
    return argument_parser().parse_args(
        [
            "--phase",
            phase,
            "--method",
            method,
            "--model-config",
            str(ROOT / "configs/slackpipe_eval/llama_8b.json"),
            "--pp",
            "2",
            "--num-stages",
            "4",
            "--microbatches",
            "4",
            "--output",
            str(tmp_path / "output"),
        ]
    )


@pytest.fixture
def forbid_gpu_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("GPU initialization/binding/work was reached before preflight")

    for name in ("set_device", "init", "mem_get_info", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    return forbidden


@pytest.mark.parametrize(
    "available,count,reason",
    [
        (False, 0, "No visible CUDA devices"),
        (False, 2, "CUDA unavailable"),
        (True, 1, "Insufficient CUDA devices"),
    ],
)
def test_test_fixture_skips_before_modulo_or_binding(
    monkeypatch, forbid_gpu_initialization, available, count, reason
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(Utils, "rank", 0)
    monkeypatch.setattr(Utils, "world_size", 2)
    monkeypatch.setattr(Utils, "inited", False)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(pytest.skip.Exception, match=reason) as error:
        Utils.initialize_distributed()
    assert f"required=2, cuda_available={available}, visible_devices={count}" in str(error.value)
    assert not Utils.inited


def test_fixture_counts_all_local_ranks_not_just_pp(monkeypatch, forbid_gpu_initialization):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    # PP=2 with TP=2 still requires four local workers/devices.
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    with pytest.raises(pytest.skip.Exception, match="required=4.*visible_devices=2"):
        Utils.require_cuda_devices()


@pytest.mark.parametrize("entry", ["launcher", "worker", "isolated"])
@pytest.mark.parametrize(
    "available,count,reason",
    [
        (False, 0, "No visible CUDA devices"),
        (False, 2, "CUDA unavailable"),
        (True, 1, "Insufficient CUDA devices"),
    ],
)
def test_production_preflight_before_gpu_work(
    tmp_path, monkeypatch, forbid_gpu_initialization, entry, available, count, reason
):
    from tools import slackpipe_eval_worker as worker
    from tools import slackpipe_isolated_profile as isolated

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    args = args_for(tmp_path)
    if entry == "launcher":
        monkeypatch.setattr(Experiment, "_execute", forbid_gpu_initialization)
        monkeypatch.setattr(Experiment, "_base_context", forbid_gpu_initialization)
        call = lambda: Experiment(args).execute()
    elif entry == "worker":
        args = normalize_args(args, worker=True)
        args.action = "benchmark"
        for name, value in dict(WORLD_SIZE=2, LOCAL_WORLD_SIZE=2, RANK=0, LOCAL_RANK=0).items():
            monkeypatch.setenv(name, str(value))
        monkeypatch.setattr(worker, "transformer_config", forbid_gpu_initialization)
        call = lambda: worker.run_worker(args)
    else:
        args = normalize_args(args)
        if available and count == 1:
            # Layer collection requires one device, regardless of target PP.
            assert require_cuda_devices(1, local_rank=0) == 1
            return
        call = lambda: isolated.collect(args)
    with pytest.raises(RuntimeError, match=reason):
        call()
    assert not args.output.exists()


@pytest.mark.parametrize("local_rank,local_world_size", [(-1, 2), (2, 2), (0, 0), (1, 1)])
def test_worker_rejects_invalid_rank_config(
    tmp_path, monkeypatch, forbid_gpu_initialization, local_rank, local_world_size
):
    from tools import slackpipe_eval_worker as worker

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    for name, value in dict(
        WORLD_SIZE=2, LOCAL_WORLD_SIZE=local_world_size, LOCAL_RANK=local_rank, RANK=local_rank
    ).items():
        monkeypatch.setenv(name, str(value))
    args = normalize_args(args_for(tmp_path), worker=True)
    args.action = "benchmark"
    monkeypatch.setattr(worker, "transformer_config", forbid_gpu_initialization)
    with pytest.raises(ValueError, match="Invalid (RANK|LOCAL_RANK|LOCAL_WORLD_SIZE)"):
        worker.run_worker(args)


@pytest.mark.parametrize("helper", [require_cuda_devices, Utils.require_cuda_devices])
def test_unknown_probe_errors_are_not_skips_or_capacity_errors(monkeypatch, helper):
    def broken():
        raise RuntimeError("unexpected probe bug")

    monkeypatch.setattr(torch.cuda, "is_available", broken)
    with pytest.raises(RuntimeError, match="unexpected probe bug"):
        helper(1)


def test_capable_fixture_does_not_hide_binding_failure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(Utils, "rank", 0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    def failure(device):
        raise RuntimeError("binding failure after successful preflight")

    monkeypatch.setattr(torch.cuda, "set_device", failure)
    with pytest.raises(RuntimeError, match="binding failure"):
        Utils.initialize_distributed()


@pytest.mark.parametrize("count", [0, 1])
def test_production_cli_exits_nonzero_before_launch(tmp_path, count):
    code = (
        # Import dependency libraries before mocking CUDA capability. TE queries
        # the real device at import time when is_available() advertises one.
        "import torch; from tools.run_slackpipe_eval import main; "
        f"torch.cuda.is_available=lambda: {bool(count)!r}; "
        f"torch.cuda.device_count=lambda: {count}; "
        "main()"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--phase",
            "benchmark",
            "--method",
            "interleaved-1f1b",
            "--model-config",
            str(ROOT / "configs/slackpipe_eval/llama_8b.json"),
            "--pp",
            "2",
            "--num-stages",
            "4",
            "--microbatches",
            "4",
            "--output",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert f"visible_devices={count}" in result.stderr
    assert "GPU execution cannot start" in result.stderr
    assert "ZeroDivisionError" not in result.stderr
    assert not (tmp_path / "output").exists()


def test_cpu_native_solve_is_not_applicable_without_probing_cuda(
    tmp_path, monkeypatch, forbid_gpu_initialization
):
    monkeypatch.setattr(torch.cuda, "is_available", forbid_gpu_initialization)
    monkeypatch.setattr(torch.cuda, "device_count", forbid_gpu_initialization)
    args = args_for(tmp_path, phase="solve")
    assert Experiment(args).ensure("solve")["status"] == "not_applicable"


def test_schedule_preflight_not_masked_by_missing_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    args = args_for(tmp_path, phase="calibrate")
    args.logical_stages = args.pp
    with pytest.raises(ValueError, match="interleav"):
        Experiment(args)


def test_torchrun_preflight_failure_terminates_waiting_peer(tmp_path):
    """CPU-only launch: rank 0 fails capability checks, rank 1 never uses CUDA."""
    from tools.slackpipe_eval_common import run

    script = tmp_path / "worker.py"
    pid_file = tmp_path / "peer.pid"
    script.write_text(
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "pid_file = Path(sys.argv[1])\n"
        "if int(os.environ['LOCAL_RANK']) == 1:\n"
        "    pid_file.write_text(str(os.getpid()))\n"
        "    time.sleep(120)\n"
        "else:\n"
        "    from tools.slackpipe_eval_common import require_cuda_devices\n"
        "    deadline = time.monotonic() + 10\n"
        "    while not pid_file.exists() and time.monotonic() < deadline:\n"
        "        time.sleep(0.05)\n"
        "    require_cuda_devices(2, local_rank=0)\n"
    )
    log = tmp_path / "torchrun.log"
    with pytest.raises(RuntimeError, match="Command failed"):
        run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                script,
                pid_file,
            ],
            log,
            30,
            dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT)),
        )
    assert "GPU execution cannot start" in log.read_text()
    assert pid_file.exists()
    # The launcher, not a new custom distributed cleanup mechanism, owns the peer.
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
