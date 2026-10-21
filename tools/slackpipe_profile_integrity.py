# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Check cost profile structure and identity without statistical acceptance rules."""

import argparse
import json
from pathlib import Path

from megatron.core.pipeline_parallel.slackpipe.profile_integrity import validate_profile_integrity


def main() -> None:
    """Validate one existing profile without changing it or collecting new data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    validate_profile_integrity(json.loads(args.profile.read_text()))
    print(f"Profile structure and identity validated: {args.profile}")


if __name__ == "__main__":
    main()
