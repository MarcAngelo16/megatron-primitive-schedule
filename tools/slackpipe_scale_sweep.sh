#!/usr/bin/env bash
# Host orchestration only. All Python/Megatron/solver execution is containerized.
# Fixture-only diagnostic. New solves use cpsat-no-hint, not the historical
# hinted joint solver; compare historical results using their recorded checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"
out="${SLACKPIPE_SCALE_OUTPUT:-slackpipe_experiments/heterogeneous_scale_sweep}"
container_root="${SLACKPIPE_CONTAINER_REPO:-/workspace/Megatron-LM}"
phase="${1:-all}"
selected="${2:-all}"
test -f "$out/runtime_checkpoint.json"

run_megatron() {
    local log="$1"
    shift
    local launcher=(/opt/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2)
    docker exec --user "$(id -u):$(id -g)" -e OMP_NUM_THREADS=1 \
        -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e HOME=/tmp/slackpipe-user \
        -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 -e PYTHONPATH="$container_root" \
        -e LOG="$container_root/$log" slackpipe-dev \
        bash -c 'exec "$@" > "$LOG" 2>&1' _ \
        "${launcher[@]}" "$@"
}

for scale in SMALL MEDIUM LARGE; do
    if [[ "$selected" != all && "$selected" != "$scale" ]]; then continue; fi
    case "$scale" in
        SMALL) hidden=128; heads=4; seq=64; lr=0.000001 ;;
        MEDIUM) hidden=256; heads=8; seq=128; lr=0.000001 ;;
        LARGE) hidden=512; heads=16; seq=256; lr=0.000000001 ;;
    esac
    dir="$out/$scale"
    docker exec --user "$(id -u):$(id -g)" slackpipe-dev mkdir -p "$container_root/$dir"
    if [[ "$phase" == all || "$phase" == calibrate ]]; then
        printf 'Calibrating %s\n' "$scale"
        run_megatron "${dir}.calibration.log" tests/unit_tests/pipeline_parallel/slackpipe_heterogeneous_calibration.py \
            --output-dir "$container_root/$dir" --hidden-size "$hidden" \
            --num-attention-heads "$heads" --seq-length "$seq" \
            --warmup-iterations 5 --iterations 10 --learning-rate "$lr"
    fi
    if [[ "$phase" == all || "$phase" == solve ]]; then
        printf 'Solving %s\n' "$scale"
        docker exec -w "$container_root" slackpipe-dev /opt/venv/bin/python \
            -m tools.slackpipe_profile_integrity --profile "$container_root/$dir/cost_profile.json"
        docker run --rm --entrypoint /bin/bash --user "$(id -u):$(id -g)" \
            -v "$root/slackpipe:/workspaces/slackpipe" -v "$root:$container_root" \
            -w /workspaces/slackpipe slackpipe-ortools-runtime:local -lc \
            "./build/ortools/slackpipe_cli --algorithm cpsat-no-hint --B 4 --N 4 --J 2 --L 12 \
            --min-layers 1 --cost-profile $container_root/$dir/cost_profile.json \
            --time-limit 60 --num-workers 1 \
            --output-prefix $container_root/$dir/joint --emit-plan $container_root/$dir/joint.plan.json" \
            > "$dir/solver.log" 2>&1
    fi
    common=(tests/unit_tests/pipeline_parallel/slackpipe_perf_benchmark.py
        --plan "$container_root/$dir/joint.plan.json"
        --heterogeneous-config "$container_root/$dir/model.json"
        --warmup-iterations 5 --iterations 50 --hidden-size "$hidden"
        --num-attention-heads "$heads" --seq-length "$seq" --micro-batch-size 1
        --vocab-size 128 --learning-rate "$lr" --strict-fp32 --disable-slackpipe-nvtx)
    if [[ "$phase" == all || "$phase" == benchmark ]]; then
        for rep in 0 1 2; do
            printf 'Benchmark %s repetition %s\n' "$scale" "$rep"
            run_megatron "$dir/benchmark_rep$rep.log" "${common[@]}" \
                --output-dir "$container_root/$dir/benchmark/rep$rep" \
                --slackpipe-transport nccl-rma --order-rotation "$rep"
        done
    fi
    if [[ ( "$phase" == all || "$phase" == transport ) && "$scale" != MEDIUM ]]; then
        # Isolated same-plan C controls avoid retained RMA groups contaminating
        # the P2P memory baseline. Alternate transport order across repetitions.
        for rep in 0 1 2; do
            transports=(nccl-p2p nccl-rma)
            if [[ "$rep" == 1 ]]; then transports=(nccl-rma nccl-p2p); fi
            for transport in "${transports[@]}"; do
                printf 'Transport %s %s repetition %s\n' "$scale" "$transport" "$rep"
                run_megatron "$dir/transport_${transport}_rep$rep.log" "${common[@]}" \
                    --output-dir "$container_root/$dir/transport/$transport/rep$rep" \
                    --slackpipe-transport "$transport" --only-mode C_slackpipe_solver_order
            done
        done
    fi
done
