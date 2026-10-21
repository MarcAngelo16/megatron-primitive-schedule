# Fork NVSkills CI Removal Audit

Date: 2026-10-21 (Asia/Tokyo). Repository:
`RioReal/megatron-primitive-schedule`, branch `slackpipe/mvp`, HEAD
`322fdf68cfcc146ec23d44b9ed2dd6ffdfc7bcaf`.

This report records the pre-publication audit. During that audit no commit,
push, PR, comment, workflow dispatch, settings/secret/variable change or upstream
modification was made. Subsequent publication was separately authorized.
The existing dirty refactor, GPU-preflight fixes and historical data remain.
The index was empty. `.github/`, `skills/`, `AGENTS.md`, CI recipes and shared
CI scripts had no pending changes at entry. The unrelated starting tracked
diff covered 123 paths; it is not part of this CI removal.

## Purpose and Removal Boundary

Deleted only `.github/workflows/request-nvskills-ci.yml` (22 lines). Its sole
job called the external reusable workflow for PR comments and trusted skill
signature pushes. It exported no workflow outputs or artifacts and had no
local downstream consumers. No SlackPipe test/build ran inside it.

Read the actual upstream definitions, not just the caller name:

- [team-request.yml](https://github.com/NVIDIA/skills/blob/main/.github/workflows/team-request.yml)
  dispatches centralized skill validation after checking requester permission,
  PR identity and watched skill paths. Its separate pull-request branch calls
  the status gate. This fork's caller did not subscribe to pull_request events.
- [require-nvskills-status.yml](https://github.com/NVIDIA/skills/blob/main/.github/workflows/require-nvskills-status.yml)
  checks the configurable `NVSkills CI` status and handles generated signature
  commits. It does not build or test Megatron/SlackPipe.
- [the pinned skills sync workflow](https://github.com/NVIDIA-NeMo/FW-CI-templates/blob/v0.91.0/.github/workflows/_sync_skills.yml)
  creates agent-directory symlinks and a CLAUDE.md link, committing them if
  needed. It is separate from centralized validation and was not executed.

The removed caller had only `contents: read` and `pull-requests: read`.
Upstream declares `statuses: read`, but no local addition of that permission
was present. No remaining permission was removed or widened.

Repository-wide searches included hidden CI paths, actions, scripts, tests,
configs and current docs, excluding Git objects and generated experiment/build
trees. Exact integration terms matched only the deleted workflow. No other
workflow_call/workflow_run, script dispatch, needs/result aggregation, output,
artifact or status consumer referenced it. The existing workflow_run listener
in `auto-swap-labels.yml` listens to `Review Trigger`, not NVSkills.

Broader skill-related hits were reviewed rather than deleted:

- Keep `AGENTS.md`, `skills/**`, inherited skill signatures, cards, evaluations
  and BENCHMARK reports. They are developer guidance and historical evidence.
  Recent skill history is inherited NVIDIA work; no current fork skill changes
  or local build/runtime consumers of those signatures were found. This does
  not claim the signatures were cryptographically revalidated.
- Keep `sync-skills.yml`, Claude review/nightly-sync skill instructions and
  their shared PAT usage. They do not call the removed validation workflow.
- Keep every other workflow, local action, container and test/build script,
  including NVIDIA infrastructure references. Existing broad permissions and
  unrelated summary logic were not changed as part of this narrow task.

No current fork usage instructions requiring `/nvskills-ci` or its dedicated
token were found. Local SlackPipe development does not require requesting this
central check. Historical NVSkills-Eval reports retain their original text.
Server-side secrets/variables, if any, were neither enumerated nor removed.

## GitHub Rules: Read-Only Findings

`gh` is not installed on host/container. No HTTP token was available from
GH_TOKEN/GITHUB_TOKEN, the existing Git HTTPS credential helper (noninteractive),
or a gh hosts file. Existing SSH Git authentication is not REST authentication.
Used GET requests only, via Python standard-library urllib; no dependencies
installed and no credentials printed.

Endpoints below are relative to
`https://api.github.com/repos/RioReal/megatron-primitive-schedule`:

| GET endpoint | Observed response |
| --- | --- |
| repository | 200; public, personal owner `RioReal`, default branch `feature/schedule` |
| `/branches?per_page=100` | 200; `slackpipe/mvp` and `feature/schedule`, both `protected: false`; no further page |
| `/rulesets?includes_parents=true&per_page=100` | 200; empty list, including visible parent rulesets; no further page |
| `/rules/branches/slackpipe%2Fmvp` | 200; empty list |
| `/rules/branches/feature%2Fschedule` | 200; empty list |
| `/branches/slackpipe%2Fmvp/protection` | 401, Requires authentication |
| `/branches/feature%2Fschedule/protection` | 401, Requires authentication |

No applicable required workflow/status rule was returned by the public ruleset
APIs. **Traditional branch-protection details and exact required check names
remain unverified**, despite the public branch summary reporting unprotected.
The 401 responses are not evidence of absent rules. Before publishing, an owner
should inspect Branch protection and Rulesets for both branches with authorized
access. If a rule requires `NVSkills CI`, a configured replacement status context,
or a check from `Request NVSkills CI`, that exact rule must be adjusted separately.
No such server-side adjustment has been made or claimed here. Deleting YAML does
not satisfy/remove a remotely required check, and no dummy success job was added.

## Retained Coverage: Before Equals After

The source includes upstream CI infrastructure, but this is **not** proof of
automatic SlackPipe coverage on this fork. `cicd-main.yml` triggers pushes only
to `pull-request/[0-9]+` and `deploy-release/*`, plus schedule, merge_group and
workflow_dispatch. Its key gate/configuration jobs require repository
`NVIDIA/Megatron-LM`. A push to `slackpipe/mvp` is not an automatic full test run.
This inventory describes the inspected dirty workspace; publishing the CI
removal does not include its unrelated pending refactor or new test files.

| Capability | Retained implementation and actual coverage boundary, unchanged |
| --- | --- |
| Python format/lint | `cicd-main.yml:linting` -> `tools/autoformat.sh`; gated upstream job. Manual container formatting remains available. |
| Python unit tests | Main workflow -> `.github/actions/action.yml` -> workload launcher -> H100 unit-test recipe -> `tests/unit_tests/run_ci_test.sh`. Recipe includes `pipeline_parallel/**/*.py`, but requires upstream infrastructure. |
| Five-method entry/config contracts | `test_slackpipe_methods.py`, eval/topology/receipt tests remain in that bucket. No independent fork-native workflow was found. |
| Stage/layer profiling, integrity, residuals | Profiling-contract, producer, isolated, cost-profile and residual tests remain in the same bucket; actual GPU collection requires its runtime prerequisites. |
| Plan parser, validator, replay | Python parser and C++ integration tests remain. Producer-to-validator tests skip if no prebuilt no-OR CLI exists. No automatic CLI build step was found in GitHub/GitLab CI. |
| No-OR and OR-enabled C++ | `slackpipe/CMakeLists.txt`, CTest suites and `slackpipe/scripts/validate_ortools_evaluation.sh` remain. Existing README/container commands are manual, not wired into a dedicated workflow. |
| GPU / PP2 / equivalence / P2P / RMA | Existing torchrun tests and artifact/evaluation entry points remain intact. Upstream recipe defaults to 8 ranks; tests requiring WORLD_SIZE=2 do not become PP2 acceptance simply by inclusion in that bucket. Dedicated manual two-rank commands remain necessary. |

Installation, build/container, functional GPU tests, coverage artifacts and
`Nemo_CICD_Test`/other result summaries are all unchanged. No assertion, rank
guard, runner, image, trigger, command, job name or success/failure rule was
weakened. Prior manually recorded test results are not relabelled CI runs.

## Static Validation

Host/container have no actionlint or yamllint; used existing container PyYAML
without installation. A `docker exec -i -w /workspace/Megatron-LM slackpipe-dev
/opt/venv/bin/python -` check parsed all retained workflows/actions with
`yaml.BaseLoader` (preserving the `on` key), checked duplicate mapping keys,
explicit dependency existence/cycles, literal local uses and local job outputs,
and compared retained `.github` file bytes to HEAD.

Results: **32 -> 31 workflows**, **2 local actions**, **93 needs edges unchanged**,
**102 -> 101 uses**, no missing literal dependency or local output introduced.
All retained tracked `.github` files are byte-identical to HEAD. Three local
action paths are supplied by prior external checkouts (install-test,
multi-approval-bot, release-docs), not missing repository files.

Two pre-existing warnings remain in `cicd-main.yml`: `Coverage` and
`cleanup-taint-node` reference `needs.pre-flight` without listing it in their
direct needs. They exist before/after, are unrelated to NVSkills, and were not
changed. Therefore this audit does not certify every retained workflow as
actionlint/GitHub-valid. External reusable-output contracts and dynamic artifact
expressions were not fully revalidated; their bytes and consumers are unchanged.
The deleted workflow had no artifacts/outputs to leave dangling.

Additional checks executed:

```bash
rg -n --hidden -i 'request-nvskills-ci|team-request\.yml|require-nvskills-status|NVIDIA/nvskills-ci|NVSkills CI|NVSKILLS_|/nvskills-ci' \
  .github .gitlab tools tests configs docker README.md AGENTS.md
# Exit 1: no active integration references after removal.
git diff -- .github
# Exactly the 22-line workflow deletion.
git diff --check
# Passed.
git diff --cached --stat
# Empty index.
```

Only this audit document and that deletion are new in this task. No shared
script or result aggregation was edited, so no Python, C++, training or GPU
suite rerun was needed. Static inspection is not a successful GitHub run.
At audit completion, the local removal had not changed GitHub. Publication
changes the branch's workflow files, not repository rules or secrets; it is not
evidence of a successful remote CI run.
