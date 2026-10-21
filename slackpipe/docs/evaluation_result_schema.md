# Result Schema v2

The C++ result includes the canonical `method_id` and its `display_name` from
the [method registry](evaluation_methods.md), including unavailable and
no-solution outcomes. Slack-based Heuristic also uses `slack-heuristic` for
`solution_source` and the nested `heuristic.method`. Existing timing/progress
fields and the `slackpipe.heuristic.v1` metadata format identifier are unchanged.

Raw solver fields, independent replay, proof scope, bounds and no-solution records are specified in the [current five-method contract](../../docs/slackpipe_five_methods.md).

C++ results use schema/method version 2; Python receipts/context use v3. Earlier hint-backed results are not migrated to the new method semantics. The refactor verification record is [here](../../docs/slackpipe_method_simplification.md).
