# Method Contracts

The C++ registry exposes these exact method IDs and display names:

| Method ID | Display Name |
| --- | --- |
| `1f1b` | 1F1B |
| `interleaved-1f1b` | Interleaved 1F1B |
| `octopipe-tuning` | OctoPipe-Tuning |
| `cpsat-no-hint` | CP-SAT without hints |
| `slack-heuristic` | Slack-based Heuristic |

`slackpipe_cli --describe-method METHOD` reports `method_id` and `display_name`
for all five methods. Only the three optimization methods run in this CLI;
the native baselines still execute in Megatron. Display names do not change
method contract hashes or search semantics.

Semantics and topology are specified in the [current five-method contract](../../docs/slackpipe_five_methods.md).

C++ results use schema/method version 2; Python receipts/context use v3. Earlier hint-backed results are not migrated to the new method semantics. The refactor verification record is [here](../../docs/slackpipe_method_simplification.md).
