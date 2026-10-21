#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/ortools}"
ORTOOLS_PREFIX="${ORTOOLS_PREFIX:-/opt/or-tools}"
ORTOOLS_DIR="${ORTOOLS_DIR:-${ORTOOLS_PREFIX}/lib/cmake/ortools}"
ABSL_DIR="${ABSL_DIR:-${ORTOOLS_PREFIX}/lib/cmake/absl}"
PROTOBUF_DIR="${PROTOBUF_DIR:-${ORTOOLS_PREFIX}/lib/cmake/protobuf}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || printf '2')}"
TEST_FILTER="${SLACKPIPE_OR_TEST_FILTER:-^slackpipe_core_ortools_smoke_tests$}"

if [[ ! -f "${ORTOOLS_DIR}/ortoolsConfig.cmake" ]]; then
  cat >&2 <<EOF
OR-Tools CMake package was not found.

Expected:
  ${ORTOOLS_DIR}/ortoolsConfig.cmake

Set either:
  ORTOOLS_PREFIX=/path/to/ortools
or:
  ORTOOLS_DIR=/path/to/ortools/lib/cmake/ortools

No OR-disabled SlackPipe binary will be produced by this script.
EOF
  exit 1
fi

cmake_args=(
  -S "${ROOT_DIR}"
  -B "${BUILD_DIR}"
  -DSLACKPIPE_ENABLE_ORTOOLS=ON
  -DSLACKPIPE_BUILD_TESTS=ON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_PREFIX_PATH="${ORTOOLS_PREFIX}"
  -Dortools_DIR="${ORTOOLS_DIR}"
)
if [[ -f "${ABSL_DIR}/abslConfig.cmake" ]]; then
  cmake_args+=(-Dabsl_DIR="${ABSL_DIR}")
fi
if [[ -f "${PROTOBUF_DIR}/protobuf-config.cmake" ]]; then
  cmake_args+=(-DProtobuf_DIR="${PROTOBUF_DIR}")
fi

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" --target slackpipe_core_tests slackpipe_cli octopipe_phase_diagnostic -j"${BUILD_JOBS}"
"${BUILD_DIR}/slackpipe_cli" --build-info
ctest --test-dir "${BUILD_DIR}" --output-on-failure -R "${TEST_FILTER}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
for method in slack-heuristic octopipe-tuning cpsat-no-hint; do
  "${BUILD_DIR}/slackpipe_cli" --algorithm "${method}" --B 2 --N 2 --J 2 --L 4 \
    --time-limit 5 --num-workers 1 --seed 123 --output-prefix "${TMP_DIR}/${method}" \
    --emit-plan "${TMP_DIR}/${method}.plan.json" > /dev/null
  "${BUILD_DIR}/slackpipe_cli" --validate-result "${TMP_DIR}/${method}.json" > /dev/null
done
"${BUILD_DIR}/slackpipe_cli" --algorithm cpsat-no-hint --B 2 --N 2 --J 2 --L 4 \
  --time-limit 5 --enforce-activation-cap true --activation-cap-units 8 \
  --output-prefix "${TMP_DIR}/capped" > /dev/null
"${BUILD_DIR}/slackpipe_cli" --validate-result "${TMP_DIR}/capped.json" > /dev/null
echo "OR-Tools five-method backend validation completed successfully."
