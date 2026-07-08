from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from imessage_mlx.utils import ensure_private_dir, read_jsonl, sha256_text, write_json, write_jsonl


def split_sessions(
    sessions_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    train_fraction: float = 0.90,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    guard_days: int = 7,
) -> dict[str, Any]:
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to one")
    sessions = list(read_jsonl(sessions_path))
    if len(sessions) < 3:
        raise ValueError("At least three complete sessions are required for train/validation/test")

    deduplicated: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for session in sorted(sessions, key=lambda row: (row["end_ns"], row["session_id"])):
        content_hash = sha256_text(str(session["text"]))
        if content_hash in deduplicated:
            duplicate_count += 1
            continue
        session["content_hash"] = content_hash
        deduplicated[content_hash] = session
    ordered = sorted(deduplicated.values(), key=lambda row: (row["end_ns"], row["session_id"]))
    count = len(ordered)
    train_end_index = max(1, min(count - 2, int(count * train_fraction)))
    validation_end_index = max(
        train_end_index + 1, min(count - 1, int(count * (train_fraction + validation_fraction)))
    )
    train_boundary = int(ordered[train_end_index]["end_ns"])
    validation_boundary = int(ordered[validation_end_index]["end_ns"])
    guard_ns = guard_days * 24 * 60 * 60 * 1_000_000_000

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    guarded = 0
    for session in ordered:
        end_ns = int(session["end_ns"])
        if guard_ns and (
            abs(end_ns - train_boundary) < guard_ns or abs(end_ns - validation_boundary) < guard_ns
        ):
            guarded += 1
            continue
        if end_ns < train_boundary:
            splits["train"].append(session)
        elif end_ns < validation_boundary:
            splits["validation"].append(session)
        else:
            splits["test"].append(session)

    if any(not values for values in splits.values()):
        raise ValueError(
            "Guard bands left an empty split; reduce guard_days only after reviewing the date range"
        )

    output_directory = ensure_private_dir(output_dir)
    for name, values in splits.items():
        write_jsonl(output_directory / f"{name}.jsonl", values)

    overlap_counter: Counter[str] = Counter()
    for name, values in splits.items():
        for value in values:
            overlap_counter[f"{name}:{value['content_hash']}"] += 1
    hash_sets = {
        name: {value["content_hash"] for value in values} for name, values in splits.items()
    }
    overlap = len(
        (hash_sets["train"] & hash_sets["validation"])
        | (hash_sets["train"] & hash_sets["test"])
        | (hash_sets["validation"] & hash_sets["test"])
    )
    report = {
        "input_sessions": len(sessions),
        "deduplicated_sessions": count,
        "removed_duplicates": duplicate_count,
        "removed_by_guard_band": guarded,
        "guard_days": guard_days,
        "counts": {name: len(values) for name, values in splits.items()},
        "date_ranges_ns": {
            name: {
                "start": min(v["start_ns"] for v in values),
                "end": max(v["end_ns"] for v in values),
            }
            for name, values in splits.items()
        },
        "cross_split_duplicate_hashes": overlap,
    }
    write_json(report_path, report)
    if overlap:
        raise RuntimeError(f"Detected {overlap} duplicate session hashes across splits")
    return report
