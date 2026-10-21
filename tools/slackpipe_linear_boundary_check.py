# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-factor diagnostic: existing non-batched native P2P, audit only.

This entry never changes the native schedule or production configuration. It
must not be used to relabel the frozen batched-P2P performance observations.
"""

from unittest.mock import patch

from tools import slackpipe_eval_worker, slackpipe_linear_worker


def main():
    original = slackpipe_eval_worker.transformer_config

    def configuration(*args, **kwargs):
        result = original(*args, **kwargs)
        result.batch_p2p_comm = False
        return result

    with patch.object(slackpipe_eval_worker, "transformer_config", configuration):
        slackpipe_linear_worker.main()


if __name__ == "__main__":
    main()
