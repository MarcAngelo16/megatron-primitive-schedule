# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Stage identity, read-only compatibility checks and auditable v1 migration."""

from pathlib import Path

from tools.slackpipe_eval_common import digest, write_json
from tools.slackpipe_eval_config import fingerprint

RECEIPT_SCHEMA = "slackpipe.eval_receipt.v4"


def artifact_error(receipt: dict, root: Path) -> str | None:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return "missing artifact checksums"
    for name, checksum in artifacts.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            return f"artifact outside experiment: {name}"
        if not path.is_file() or digest(path) != checksum:
            return f"missing/changed artifact: {name}"
    for key in ("profile", "plan", "summary", "solver_result"):
        if key in receipt and receipt[key] not in artifacts:
            return f"unbound {key} artifact"
    return None


def receipt_valid(receipt: dict, context: dict, root: Path) -> bool:
    """Exact context/byte check, also used by the offline table exporter."""
    return (
        receipt.get("schema_version") == RECEIPT_SCHEMA
        and fingerprint(receipt.get("context")) == fingerprint(context)
        and receipt.get("context_hash", fingerprint(context)) == fingerprint(context)
        and artifact_error(receipt, root) is None
    )


def parent_identity(receipt: dict) -> str:
    """Migration notes, aliases and timestamps must not invalidate scientific outputs."""
    return fingerprint({k: receipt[k] for k in ("stage", "context", "artifacts")})


def archive_receipt(path: Path, receipt: dict) -> None:
    archive = path.parent / "history" / f"{path.stem}.{fingerprint(receipt)}.json"
    if not archive.exists():
        write_json(archive, receipt)


def context_differences(old: dict, new: dict, prefix: str = "") -> list[str]:
    changed = []
    for key in sorted(set(old) | set(new)):
        label = f"{prefix}{key}"
        if key not in old or key not in new:
            changed.append(label)
        elif isinstance(old[key], dict) and isinstance(new[key], dict):
            changed.extend(context_differences(old[key], new[key], label + "."))
        elif fingerprint(old[key]) != fingerprint(new[key]):
            changed.append(label)
    return changed


def assess_receipt(receipt: dict, context: dict, root: Path, parents: dict) -> tuple:
    """Return (classification, explanation, accepted receipt), without filesystem writes."""
    if receipt.get("status") not in ("passed", "no_valid_solution"):
        return "invalid", f"receipt status is {receipt.get('status')}", None
    if error := artifact_error(receipt, root):
        return "invalid", error, None
    schema = receipt.get("schema_version")
    if schema == RECEIPT_SCHEMA:
        if (
            not isinstance(receipt.get("context"), dict)
            or receipt["context"].get("context_schema_version") != 4
        ):
            return "invalid", "unknown/missing context schema", None
        if receipt.get("context_hash") != fingerprint(receipt.get("context")):
            return "invalid", "context hash mismatch", None
        if receipt.get("stage") != context["stage"]:
            return "invalid", "receipt stage mismatch", None
        changed = context_differences(receipt["context"], context)
        if changed:
            return "stale", "changed stage inputs: " + ", ".join(changed), None
        return "compatible", "stage inputs and artifact checksums match", receipt
    return (
        "stale",
        f"historical receipt schema {schema}; v4 requires explicit profiling policy identity",
        None,
    )
