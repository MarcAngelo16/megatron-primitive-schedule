# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Strict research model specifications and allocation-free exact shape counts.

The count mirrors GPT TE and hybrid_stack_spec at TP=1, untied embeddings,
RMSNorm, bias-free projections; Mamba retains its depthwise convolution bias.
Unsupported architecture options are rejected rather than approximately counted.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "slackpipe.eval_model.v1"
METHOD_CONTRACT_VERSION = 3


@dataclass(frozen=True)
class Method:
    method_id: str
    display_name: str
    executor: str
    requires_solve: bool = False
    requires_ortools: bool = False
    backend: str | None = None
    fixed_flags: tuple[str, ...] = ()
    supports_threads: bool = False
    supports_activation_cap: bool = False
    topology: str = "explicit_cyclic"
    budget_policy: str = "not_applicable"

    def contract(self) -> dict:
        return dict(version=METHOD_CONTRACT_VERSION, **asdict(self))


METHODS = {
    m.method_id: m
    for m in (
        Method("1f1b", "Native 1F1B", "megatron-native", topology="N=PP"),
        Method(
            "interleaved-1f1b",
            "Native interleaved 1F1B",
            "megatron-native",
            topology="PP>1,N>PP,B%PP=0",
        ),
        Method(
            "octopipe-tuning",
            "OctoPipe-Tuning",
            "slackpipe",
            True,
            backend="octopipe-tuning",
            budget_policy="wall_clock",
            supports_threads=False,
        ),
        Method(
            "cpsat-no-hint",
            "CP-SAT without hints",
            "slackpipe",
            True,
            True,
            "cpsat-no-hint",
            supports_threads=True,
            supports_activation_cap=True,
            budget_policy="wall_clock_no_external_incumbent_or_fallback",
        ),
        Method(
            "slack-heuristic",
            "Slack-based Heuristic",
            "slackpipe",
            True,
            backend="slack-heuristic",
            budget_policy="wall_clock_coupled",
        ),
    )
}
SCHEDULES = COMPARISON_SCHEDULES = tuple(METHODS)
PLAN_SCHEDULES = tuple(m.method_id for m in METHODS.values() if m.requires_solve)


def canonical_schedule(schedule: str) -> str:
    """Retired names must not silently reuse incompatible receipts."""
    if schedule not in METHODS:
        raise ValueError(f"Unknown/retired method {schedule!r}; choose {', '.join(METHODS)}")
    return schedule


def execution_mode(schedule: str) -> str:
    """The plan executor name is not an optimization method."""
    return "slackpipe" if METHODS[canonical_schedule(schedule)].requires_solve else "baseline"


def solver_arguments(
    method_id: str, seconds: float, workers: int = 1, activation_cap: list[int] | None = None
) -> list[str]:
    method = METHODS[canonical_schedule(method_id)]
    if not method.requires_solve:
        raise ValueError(f"{method_id} is a native Megatron method")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Search wall-clock budget must be positive and finite")
    if workers < 1 or (workers != 1 and not method.supports_threads):
        raise ValueError(f"{method_id} supports only one solver worker")
    if activation_cap is not None and not method.supports_activation_cap:
        raise ValueError(f"{method_id} cannot enforce activation caps")
    flags = [
        "--algorithm",
        method.backend,
        "--time-limit",
        str(seconds),
        "--num-workers",
        str(workers),
        *method.fixed_flags,
    ]
    if activation_cap is not None:
        if not activation_cap or any(v < 0 for v in activation_cap):
            raise ValueError("Invalid activation cap")
        flags += [
            "--enforce-activation-cap",
            "true",
            "--activation-cap-units",
            ",".join(map(str, activation_cap)),
        ]
    return flags


def fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_model(value: dict) -> dict:
    required = (
        "schema_version name model_family scale_label official_or_scaled architecture_source "
        "num_layers hidden_size ffn_hidden_size num_attention_heads num_query_groups vocab_size "
        "max_sequence_length normalization activation position_embedding attention_type ffn_type "
        "expected_parameter_scale defaults scaling_rule"
    ).split()
    missing = sorted(set(required) - value.keys())
    if missing:
        raise ValueError(f"Missing model fields: {missing}")
    if value["schema_version"] != SCHEMA:
        raise ValueError("Unsupported model schema")
    for key in required:
        if value[key] is None or value[key] == "":
            raise ValueError(f"Empty model field: {key}")
    for key in (
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "vocab_size",
        "max_sequence_length",
        "expected_parameter_scale",
    ):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"Expected positive integer: {key}")
    h, heads, groups = (
        value[k] for k in ("hidden_size", "num_attention_heads", "num_query_groups")
    )
    if h % heads or heads % groups:
        raise ValueError("Incompatible attention dimensions")
    if value["normalization"] != "RMSNorm" or value["attention_type"] != "GQA":
        raise ValueError("Only bias-free RMSNorm/GQA adapters are defined")
    family = value["model_family"]
    expected = {
        "llama": ("silu", "rope", "SwiGLU"),
        "nemotron_h": ("squared_relu", "none", "ReLU2"),
    }
    if (
        family not in expected
        or tuple(value[k] for k in ("activation", "position_embedding", "ffn_type"))
        != expected[family]
    ):
        raise ValueError("Unsupported model family/activation/position/FFN combination")
    if value["official_or_scaled"] not in ("scaled_research", "official_architecture"):
        raise ValueError("Invalid official/scaled designation")
    if family == "llama" and value["official_or_scaled"] != "scaled_research":
        raise ValueError("LLaMA-style presets are research architectures, not released checkpoints")
    if family == "nemotron_h":
        pattern = value.get("hybrid_pattern", "")
        if len(pattern) != value["num_layers"] or set(pattern) != set("M*-"):
            raise ValueError("Hybrid sequence must specify every M/attention/MLP global block")
        for key in ("mamba_state_dim", "mamba_head_dim", "mamba_num_groups", "mamba_num_heads"):
            if type(value.get(key)) is not int or value[key] <= 0:
                raise ValueError(f"Missing/invalid {key}")
        if value["mamba_num_heads"] % value["mamba_num_groups"]:
            raise ValueError("Mamba heads must be divisible by groups")
        if value["official_or_scaled"] == "official_architecture":
            from megatron.core.pipeline_parallel.slackpipe.hybrid import (
                NEMOTRON_H_8B_PATTERN,
                NEMOTRON_H_8B_SOURCE,
            )

            exact = dict(
                num_layers=52,
                hidden_size=4096,
                ffn_hidden_size=21504,
                num_attention_heads=32,
                num_query_groups=8,
                vocab_size=131072,
                max_sequence_length=8192,
                mamba_state_dim=128,
                mamba_head_dim=64,
                mamba_num_groups=8,
                mamba_num_heads=128,
                hybrid_pattern=NEMOTRON_H_8B_PATTERN,
                architecture_source=NEMOTRON_H_8B_SOURCE,
            )
            if any(value[k] != v for k, v in exact.items()):
                raise ValueError(
                    "Official designation requires exact authoritative Base-8K architecture"
                )
    defaults = value["defaults"]
    for key in ("seq_length", "micro_batch_size", "global_batch_size"):
        if type(defaults.get(key)) is not int or defaults[key] < 1:
            raise ValueError(f"Invalid default {key}")
    if defaults["seq_length"] > value["max_sequence_length"] or defaults.get("precision") not in (
        "fp32",
        "bf16",
    ):
        raise ValueError("Invalid default sequence/precision")
    return value


def load_model(path: Path) -> dict:
    return validate_model(json.loads(Path(path).read_text()))


def parameter_breakdown(model: dict) -> dict:
    """Count all trainable scalar elements, including norms and Mamba vectors."""
    validate_model(model)
    h, f = model["hidden_size"], model["ffn_hidden_size"]
    heads, groups = model["num_attention_heads"], model["num_query_groups"]
    attention = h * (h + 2 * groups * (h // heads)) + h * h + h
    mlp = h * f * (3 if model["ffn_type"] == "SwiGLU" else 2) + h
    if model["model_family"] == "llama":
        layers = [attention + mlp] * model["num_layers"]
    else:
        n = model["mamba_num_heads"]
        inner = n * model["mamba_head_dim"]
        conv = inner + 2 * model["mamba_num_groups"] * model["mamba_state_dim"]
        mamba = h * (inner + conv + n) + h + conv * 5 + 3 * n + inner + inner * h
        layers = [{"M": mamba, "*": attention, "-": mlp}[c] for c in model["hybrid_pattern"]]
    embedding = h * model["vocab_size"]
    total = 2 * embedding + h + sum(layers)
    target = model["expected_parameter_scale"]
    return dict(
        exact_parameter_count=total,
        target_parameter_count=target,
        difference_from_target=total - target,
        relative_difference=(total - target) / target,
        embedding=embedding,
        output=embedding,
        final_norm=h,
        layers=layers,
    )


def schedule_topology(schedule: str, pp: int, stages: int | None, microbatches: int) -> dict:
    schedule = canonical_schedule(schedule)
    if schedule not in SCHEDULES or pp not in (1, 2, 4) or microbatches < 1:
        raise ValueError("Unsupported schedule, PP or B")
    if stages is None and schedule != "1f1b":
        raise ValueError("Specify --num-stages explicitly for interleaving/plan methods")
    stages = pp if stages is None else stages
    if stages < pp or stages % pp:
        raise ValueError("Cyclic placement requires N % PP == 0")
    if schedule == "1f1b" and stages != pp:
        raise ValueError("Native 1F1B requires N=PP and VPP=None")
    if schedule == "interleaved-1f1b" and (stages == pp or pp == 1):
        raise ValueError("Native interleaving requires physical PP>1 and N>PP")
    if schedule == "interleaved-1f1b" and microbatches % pp:
        raise ValueError("Native interleaving requires B divisible by PP")
    return dict(
        pp=pp,
        num_stages=stages,
        vpp=None if stages == pp else stages // pp,
        num_microbatches=microbatches,
        tp=1,
        dp=1,
        cp=1,
    )


def transformer_config(model: dict, topology: dict, precision: str, layout=None):
    import torch

    from megatron.core.activations import squared_relu
    from megatron.core.pipeline_parallel.slackpipe.hybrid import bind_hybrid_config
    from megatron.core.transformer.transformer_config import TransformerConfig

    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    config = TransformerConfig(
        **{
            k: model[k]
            for k in (
                "num_layers",
                "hidden_size",
                "ffn_hidden_size",
                "num_attention_heads",
                "num_query_groups",
            )
        },
        **{k: v for k, v in model.items() if k.startswith("mamba_")},
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        activation_func=(
            torch.nn.functional.silu if model["model_family"] == "llama" else squared_relu
        ),
        gated_linear_unit=model["model_family"] == "llama",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pipeline_model_parallel_size=topology["pp"],
        virtual_pipeline_model_parallel_size=topology["vpp"],
        pipeline_model_parallel_layout=layout,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=precision == "bf16",
        use_cpu_initialization=True,
        batch_p2p_comm=True,
        overlap_p2p_comm=False,
    )
    if model["model_family"] == "nemotron_h":
        bind_hybrid_config(config, model["hybrid_pattern"])
    return config
