<div align="center">

Megatron-LM and Megatron Core
=============================

<h4>GPU-optimized library for training transformer models at scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.15.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

<div align="left">

## SlackPipe Integration

This research fork retains the earlier Megatron primitive-schedule work and the
current SlackPipe plan runtime. NVIDIA/Megatron attribution, licenses and upstream
documentation below are preserved.

The public evaluation methods are now exactly `1f1b`, `interleaved-1f1b`,
`octopipe-tuning`, `cpsat-no-hint`, and `slack-heuristic`.
The first two execute native Megatron schedules; the others export validated
plans for the same `pipeline_schedule=slackpipe` runtime.
Native 1F1B has N=PP. All other methods use an explicit N, with cyclic placement.
Old `slackpipe`, refined and partition-only method aliases are rejected.

**Current commands and contracts:** [Five-method evaluation](docs/slackpipe_five_methods.md).
**Refactor validation:** [Implementation review](docs/slackpipe_method_simplification.md).

### Repository Layout

- `megatron/core/pipeline_parallel/slackpipe/`: plan parsing, PP/VPP execution,
  P2P/experimental RMA, manifest/cost profiles, structural integrity and tracing.
- `slackpipe/`: tracked C++ optimizer/evaluator/exporter in this monorepo,
  not a separate nested Git repository.
- `tools/run_slackpipe_eval.py`: single staged experiment entry point.
- `tools/slackpipe_eval_config.py`: unique Python five-method registry.
- `tools/slackpipe_training.py`, `tools/slackpipe_model_support.py`: shared
  model/training implementation, with no imports from test implementations.
- `tools/run_slackpipe_real_system_campaign.py`: thin rotating batch wrapper.
- `configs/slackpipe_eval/`: explicit homogeneous and hybrid model specifications.
- `tests/unit_tests/pipeline_parallel/` and `slackpipe/tests/`: correctness tests.
- Generated build trees, experiments, profiler captures and paper builds remain
  ignored; historical experiment receipts are not rewritten.

### Capabilities and Limits

The runtime retains PP=1/PP=2 logical stages, nonuniform partitions, calibrated
compute costs, compatible heterogeneous sequential decoders and exact contiguous
global-layer ranges. PP=4/hybrid support remains subject to the existing GPU
validation record; it is not revalidated by this cleanup. TP/DP/CP, distributed
optimizer, recomputation, CUDA graphs and RMA multi-node are not added here.
Experimental `nccl-rma` retains persistent lifecycle management; `nccl-p2p`
remains the ordinary transport. BF16/large-model validation cannot be inferred
from FP32 tiny fixtures.

`slackpipe.plan.v1/v2`, `slackpipe.cost_profile.v1/v2`, and
`slackpipe.model_manifest.v1` retain their meanings. Plan v2 binds heterogeneous
global ranges to profile/manifest hashes. Current C++ result schema v2 and
Python receipt/context v4 distinguish method from executor, raw solver bounds
from independent workload bounds, predicted F/B makespan from GPU step time,
and search budget from actual elapsed time.

The profiling flow is:

```
sequential stage + layer collection by default (separate raw artifacts)
  -> stage: one initial-model residual mask and at most one refit per phase
  -> layer: median of all legal samples, no filtering
  -> structural/configuration checks; select stage by default
  -> C++ plan generator
  -> independent replay + plan/provenance validation
  -> Megatron smoke, benchmark and separate trace
```

Use `--phase`, `--method`, `--num-stages`, and
`--profiling-method {stage,layer,both}`. Omitted `--solver-profile` selects stage
for stage/both collection and layer for layer-only collection. Native standalone
benchmarks require neither profiling nor a solver. There are no statistical
stability gates, automatic retries or implicit backend fallback. A failed
nonselected backend is reported as partial; a failed selected backend blocks
downstream work. See [Profiling contract](docs/slackpipe_profiling.md) for formulas,
artifact reuse, measurement scope and minimal commands.

Historical Nemotron accuracy and paper results are not revalidated. The current
container reports CUDA unavailable and zero devices, with an NVML initialization
error; CPU checks do not establish GPU numerical equivalence or performance.
Runtime traces/NVTX/plotting retain raw capture identities.

### Container Quick Start

Use the existing `slackpipe-dev` container, checkout `/workspace/Megatron-LM`,
Python `/opt/venv/bin/python`. Do not install Megatron dependencies on the host.

```bash
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake -S slackpipe -B slackpipe/build/no-or -G Ninja -DSLACKPIPE_ENABLE_ORTOOLS=OFF
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  cmake --build slackpipe/build/no-or -j 4
docker exec -w /workspace/Megatron-LM slackpipe-dev \
  ctest --test-dir slackpipe/build/no-or --output-on-failure
docker exec -w /workspace/Megatron-LM slackpipe-dev /opt/venv/bin/python \
  -m tools.slackpipe_artifact_smoke --method slack-heuristic --output /tmp/slackpipe-new-smoke
```

The [five-method guide](docs/slackpipe_five_methods.md) gives the OR-Tools image
build, all five commands, shared-budget/cost rules, targeted Python/GPU tests,
no-solution behavior, trace plotting and resume rules. No-OR builds support the
heuristic and OctoPipe; CP-SAT explicitly reports unavailable. CPU-only smoke
does not constitute a GPU training or performance result.

## About

This repository contains two components: **Megatron-LM** and **Megatron Core**.

**Megatron-LM** is a reference example that includes Megatron Core plus pre-configured training scripts. Best for research teams, learning distributed training, and quick experimentation.

**Megatron Core** is a composable library with GPU-optimized building blocks for custom training frameworks. It provides transformer building blocks, advanced parallelism strategies (TP, PP, DP, EP, CP), mixed precision support (FP16, BF16, FP8, FP4), and model architectures. Best for framework developers and ML engineers building custom training pipelines.

**[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** provides bidirectional Hugging Face ↔ Megatron checkpoint conversion with production-ready recipes.

## Getting Started

**Install from PyPI:**

```bash
uv pip install megatron-core
```

**Or clone and install from source:**

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install -e .
```

> **Note:** Building from source can use a lot of memory. If the build runs out of memory, limit parallel compilation jobs by setting `MAX_JOBS` (e.g. `MAX_JOBS=4 uv pip install -e .`).

For NGC container setup and all installation options, see the **[Installation Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)**.

- **[Your First Training Run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)** - End-to-end training examples with data preparation
- **[Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)** - Scale training across GPUs with TP, PP, DP, EP, and CP
- **[Contribution Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)** - How to contribute to Megatron Core

# Latest News

- **[2026/03]** **Deprecating Python 3.10 support:** We're officially dropping Python 3.10 support with the upcoming 0.17.0 release. Downstream applications must raise their lower boundary to 3.12 to stay compatible with MCore.
- **[2026/01]** **[Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)** - Up to 1.48x speedup for variable-length sequence training with adaptive CP sizing.
- **[2025/12]** **Megatron Core development has moved to GitHub!** All development and CI now happens in the open. We welcome community contributions.
- **[2025/10]** **[Megatron Dev Branch](https://github.com/NVIDIA/Megatron-LM/tree/dev)** - early access branch with experimental features.
- **[2025/10]** **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints, featuring production-ready recipes for popular models.
- **[2025/08]** **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Comprehensive roadmap for MoE features including DeepSeek-V3, Qwen3, advanced parallelism strategies, FP8 optimizations, and Blackwell performance enhancements.
- **[2025/08]** **[GPT-OSS Model](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Advanced features including YaRN RoPE scaling, attention sinks, and custom activation functions are being integrated into Megatron Core.
- **[2025/06]** **[Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo)** - Best practices and optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models with performance benchmarking and checkpoint conversion tools.
- **[2025/05]** Megatron Core v0.11.0 brings new capabilities for multi-data center LLM training ([blog](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/)).

<details>
<summary>Previous News</summary>

- **[2024/07]** Megatron Core v0.7 improves scalability and training resiliency and adds support for multimodal training ([blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-Megatron-Core-functionalities/)).
- **[2024/06]** Megatron Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models](https://arxiv.org/pdf/2406.07887) and [code example](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).
- **[2024/01 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron Core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) in this repository. Megatron Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs.

</details>

# Project Structure

```
Megatron-LM/
├── megatron/
│   ├── core/                    # Megatron Core (kernels, parallelism, building blocks)
│   │   ├── models/              # Transformer models
│   │   ├── transformer/         # Transformer building blocks
│   │   ├── tensor_parallel/     # Tensor parallelism
│   │   ├── pipeline_parallel/   # Pipeline parallelism
│   │   ├── distributed/         # Distributed training (FSDP, DDP)
│   │   ├── optimizer/           # Optimizers
│   │   ├── datasets/            # Dataset loaders
│   │   ├── inference/           # Inference engines and server
│   │   └── export/              # Model export (e.g. TensorRT-LLM)
│   ├── training/                # Training scripts
│   ├── legacy/                  # Legacy components
│   ├── post_training/           # Post-training (quantization, distillation, pruning, etc.)
│   └── rl/                      # Reinforcement learning (RLHF, etc.)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

# Performance Benchmarking

For our latest performance benchmarking results, please refer to [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html).

Our codebase efficiently trains models from 2B to 462B parameters across thousands of GPUs, achieving up to **47% Model FLOP Utilization (MFU)** on H100 clusters.

![Model table](images/model_table.png)

**Benchmark Configuration:**

- **Vocabulary size**: 131,072 tokens
- **Sequence length**: 4096 tokens
- **Model scaling**: Varied hidden size, attention heads, and layers to achieve target parameter counts
- **Communication optimizations**: Fine-grained overlapping with DP (`--overlap-grad-reduce`, `--overlap-param-gather`), TP (`--tp-comm-overlap`), and PP (enabled by default)

**Key Results:**

- **6144 H100 GPUs**: Successfully benchmarked 462B parameter model training
- **Superlinear scaling**: MFU increases from 41% to 47-48% with model size
- **End-to-end measurement**: Throughputs include all operations (data loading, optimizer steps, communication, logging)
- **Production ready**: Full training pipeline with checkpointing and fault tolerance
- *Note: Performance results measured without training to convergence*

## Weak Scaling Results

Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.

![Weak scaling](images/weak_scaling.png)

## Strong Scaling Results

We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.

![Strong scaling](images/strong_scaling.png)

# Roadmaps

- **[MoE Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - DeepSeek-V3, Qwen3, advanced parallelism, FP8 optimizations, and Blackwell enhancements

# Resources

## Getting Help

- 📖 **[Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)** - Official documentation
- 🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug reports and feature requests

## Contributing

We ❤️ contributions! Ways to contribute:

- 🐛 **Report bugs** - Help us improve reliability
- 💡 **Suggest features** - Shape the future of Megatron Core
- 📝 **Improve docs** - Make Megatron Core more accessible
- 🔧 **Submit PRs** - Contribute code improvements

**→ [Contributing Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)**

## Citation

If you use Megatron in your research or project, we appreciate that you use the following citations:

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```
