cmake_minimum_required(VERSION 3.24)
project(SlackPipe LANGUAGES CXX)

option(SLACKPIPE_BUILD_TESTS "Build SlackPipe tests" ON)
set(SLACKPIPE_ENABLE_ORTOOLS "AUTO" CACHE STRING "Enable OR-Tools CP-SAT support: AUTO, ON, or OFF")
set_property(CACHE SLACKPIPE_ENABLE_ORTOOLS PROPERTY STRINGS AUTO ON OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Git QUIET)
set(SLACKPIPE_BUILD_GIT_COMMIT "")
set(SLACKPIPE_BUILD_GIT_DIRTY "")
set(SLACKPIPE_ORTOOLS_VERSION "")
if(Git_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE SLACKPIPE_BUILD_GIT_COMMIT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} status --porcelain
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE SLACKPIPE_BUILD_GIT_STATUS
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(SLACKPIPE_BUILD_GIT_STATUS STREQUAL "")
    set(SLACKPIPE_BUILD_GIT_DIRTY "false")
  else()
    set(SLACKPIPE_BUILD_GIT_DIRTY "true")
  endif()
endif()

add_library(slackpipe_warnings INTERFACE)
target_compile_options(slackpipe_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)

add_library(slackpipe_core
  src/benchmark.cc
  src/activation_cpsat.cc
  src/activation_analyzer.cc
  src/alternating_solver.cc
  src/bfs_solver.cc
  src/breadth_first.cc
  src/cost_profile.cc
  src/dag_evaluator.cc
  src/deadline.cc
  src/evaluation_method.cc
  src/fixed_order_partition_solver.cc
  src/hybrid_slack.cc
  src/instance.cc
  src/interleaving_stats.cc
  src/io.cc
  src/joint_solver.cc
  src/one_f_one_b.cc
  src/operation.cc
  src/octopipe_tuner.cc
  src/workload_bound.cc
  src/plan_export.cc
  src/pressure_pruning.cc
  src/heuristic_hint.cc
  src/result_schema.cc
  src/result_validator.cc
  src/slackpipe_solver.cc
)
target_include_directories(slackpipe_core PUBLIC include)
target_link_libraries(slackpipe_core PUBLIC slackpipe_warnings)
target_compile_definitions(slackpipe_core PUBLIC
  SLACKPIPE_BUILD_GIT_COMMIT="${SLACKPIPE_BUILD_GIT_COMMIT}"
  SLACKPIPE_BUILD_GIT_DIRTY="${SLACKPIPE_BUILD_GIT_DIRTY}"
)

string(TOUPPER "${SLACKPIPE_ENABLE_ORTOOLS}" SLACKPIPE_ENABLE_ORTOOLS_NORMALIZED)
if(NOT SLACKPIPE_ENABLE_ORTOOLS_NORMALIZED MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "SLACKPIPE_ENABLE_ORTOOLS must be AUTO, ON, or OFF")
endif()

set(SLACKPIPE_ORTOOLS_FOUND OFF)
if(NOT SLACKPIPE_ENABLE_ORTOOLS_NORMALIZED STREQUAL "OFF")
  find_package(ortools CONFIG QUIET)
  if(ortools_FOUND)
    set(SLACKPIPE_ORTOOLS_FOUND ON)
  elseif(SLACKPIPE_ENABLE_ORTOOLS_NORMALIZED STREQUAL "ON")
    message(FATAL_ERROR
      "SLACKPIPE_ENABLE_ORTOOLS=ON requires the OR-Tools CMake package. "
      "Set ortools_DIR=/path/to/ortools/lib/cmake/ortools or "
      "CMAKE_PREFIX_PATH=/path/to/ortools. Bundled OR-Tools archives may also "
      "need absl_DIR=/path/to/ortools/lib/cmake/absl and "
      "Protobuf_DIR=/path/to/ortools/lib/cmake/protobuf; "
      "scripts/validate_ortools_evaluation.sh passes these automatically.")
  endif()
endif()

if(SLACKPIPE_ORTOOLS_FOUND)
  if(DEFINED ortools_VERSION)
    set(SLACKPIPE_ORTOOLS_VERSION "${ortools_VERSION}")
  endif()
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_LIBRARIES ortools::ortools)
  check_cxx_source_compiles("
    #include \"ortools/sat/cp_model.h\"
    int main() {
      operations_research::sat::CpModelBuilder builder;
      operations_research::sat::IntVar start =
          builder.NewIntVar(operations_research::Domain(0, 10));
      operations_research::sat::IntVar size =
          builder.NewIntVar(operations_research::Domain(0, 10));
      operations_research::sat::IntVar end =
          builder.NewIntVar(operations_research::Domain(0, 10));
      operations_research::sat::IntervalVar interval =
          builder.NewIntervalVar(start, size, end);
      operations_research::sat::CumulativeConstraint cumulative =
          builder.AddCumulative(10);
      cumulative.AddDemand(interval, 1);
      return 0;
    }
  " SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED_CHECK)
  check_cxx_source_compiles("
    #include \"ortools/sat/cp_model.h\"
    int main() {
      operations_research::sat::CpModelBuilder builder;
      operations_research::sat::IntVar start =
          builder.NewIntVar(operations_research::Domain(0, 10));
      operations_research::sat::IntVar size =
          builder.NewIntVar(operations_research::Domain(0, 10));
      operations_research::sat::IntVar end =
          builder.NewIntVar(operations_research::Domain(0, 10));
      operations_research::sat::IntervalVar interval =
          builder.NewIntervalVar(start, size, end);
      operations_research::sat::CumulativeConstraint cumulative =
          builder.AddCumulative(10);
      operations_research::sat::LinearExpr demand = 2 * size;
      cumulative.AddDemand(interval, demand);
      return 0;
    }
  " SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED_CHECK)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED_CHECK)
    set(SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED 1)
  else()
    set(SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED 0)
  endif()
  if(SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED_CHECK)
    set(SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED 1)
  else()
    set(SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED 0)
  endif()
  target_sources(slackpipe_core PRIVATE src/bfs_solver_cpsat.cc)
  target_sources(slackpipe_core PRIVATE src/joint_solver_cpsat.cc)
  target_sources(slackpipe_core PRIVATE src/slackpipe_solver_cpsat.cc)
  target_compile_definitions(slackpipe_core PUBLIC SLACKPIPE_HAVE_ORTOOLS=1)
  target_link_libraries(slackpipe_core PUBLIC ortools::ortools)
else()
  target_compile_definitions(slackpipe_core PUBLIC SLACKPIPE_HAVE_ORTOOLS=0)
  set(SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED 0)
  set(SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED 0)
endif()

set(SLACKPIPE_HAVE_CUMULATIVE ${SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED})
set(SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND
    ${SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED})

if(SLACKPIPE_ORTOOLS_FOUND)
  set(SLACKPIPE_ORTOOLS_STATUS ON)
else()
  set(SLACKPIPE_ORTOOLS_STATUS OFF)
endif()
target_compile_definitions(slackpipe_core PUBLIC
  SLACKPIPE_ORTOOLS_VERSION="${SLACKPIPE_ORTOOLS_VERSION}"
  SLACKPIPE_HAVE_CUMULATIVE=${SLACKPIPE_HAVE_CUMULATIVE}
  SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND=${SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND}
  SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED=${SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED}
  SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED=${SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED}
)
message(STATUS "SlackPipe OR-Tools support: ${SLACKPIPE_ORTOOLS_STATUS}")

add_executable(slackpipe_cli apps/slackpipe_cli.cc)
target_link_libraries(slackpipe_cli PRIVATE slackpipe_core)

add_executable(slackpipe_benchmark apps/slackpipe_benchmark.cc)
target_link_libraries(slackpipe_benchmark PRIVATE slackpipe_core)

add_executable(octopipe_phase_diagnostic apps/octopipe_phase_diagnostic.cc)
target_link_libraries(octopipe_phase_diagnostic PRIVATE slackpipe_core)

if(SLACKPIPE_BUILD_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  add_executable(slackpipe_core_tests
    tests/benchmark_tests.cc
    tests/core_tests.cc
    tests/octopipe_tests.cc
    tests/octopipe_diagnostic_tests.cc
    tests/workload_bound_tests.cc
    tests/heuristic_tests.cc
  )
  if(GTest_FOUND)
    target_link_libraries(slackpipe_core_tests PRIVATE slackpipe_core GTest::gtest_main)
  else()
    target_sources(slackpipe_core_tests PRIVATE third_party/gtest/gtest_main.cc)
    target_include_directories(slackpipe_core_tests PRIVATE third_party/gtest)
    target_link_libraries(slackpipe_core_tests PRIVATE slackpipe_core)
  endif()
  add_test(NAME slackpipe_core_tests COMMAND slackpipe_core_tests)
  add_test(
    NAME slackpipe_core_ortools_smoke_tests
    COMMAND slackpipe_core_tests --gtest_filter=OrTools*)
  set_tests_properties(slackpipe_core_ortools_smoke_tests
                       PROPERTIES LABELS ortools)
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME focused_eval_runner_tests
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/focused_eval_runner_test.py
    )
    add_test(
      NAME result_validator_cli_tests
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/result_validator_cli_test.py $<TARGET_FILE:slackpipe_cli>
    )
    add_test(
      NAME heuristic_cli_tests
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/heuristic_cli_test.py
              $<TARGET_FILE:slackpipe_cli>
    )
  endif()
endif()
