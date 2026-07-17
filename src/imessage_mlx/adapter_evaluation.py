from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from imessage_mlx.adapter_worker import multiset_jaccard, normalized_words
from imessage_mlx.utils import read_jsonl, write_json

NEGATIONS = frozenset(
    {
        "ain't",
        "aint",
        "cannot",
        "can't",
        "cant",
        "couldn't",
        "couldnt",
        "didn't",
        "didnt",
        "doesn't",
        "doesnt",
        "don't",
        "dont",
        "idk",
        "idt",
        "idts",
        "isn't",
        "isnt",
        "nah",
        "never",
        "nevermind",
        "no",
        "nope",
        "not",
        "nvm",
        "shouldn't",
        "shouldnt",
        "wasn't",
        "wasnt",
        "weren't",
        "werent",
        "won't",
        "wont",
        "wouldn't",
        "wouldnt",
    }
)
NUMBER_PATTERN = re.compile(
    r"(?<![a-z0-9])(\d[\d,]*(?::\d{2})?(?:\.\d+)?)([kmb])?",
    re.IGNORECASE,
)
KEY_METRICS = (
    "prediction_words_from_source",
    "source_words_retained",
    "source_jaccard",
    "target_jaccard",
    "exact_source_copy_rate",
    "near_source_copy_rate",
    "source_numeral_retention_rate",
    "source_negation_agreement_rate",
)


def _overlap_count(left: Sequence[str], right: Sequence[str]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def _has_negation(tokens: Sequence[str]) -> bool:
    return any(token in NEGATIONS for token in tokens)


def normalized_numbers(text: str) -> Counter[str]:
    values: Counter[str] = Counter()
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    for match in NUMBER_PATTERN.finditer(text):
        raw, suffix = match.groups()
        if ":" in raw:
            hours, minutes = raw.split(":", maxsplit=1)
            canonical = str(int(hours)) if minutes == "00" else f"{int(hours)}:{minutes}"
        else:
            value = Decimal(raw.replace(",", ""))
            if suffix:
                value *= multipliers[suffix.lower()]
            canonical = format(value.normalize(), "f")
        values[canonical] += 1
    return values


def _transformation_bucket(reference_similarity: float) -> str:
    if reference_similarity < 0.5:
        return "strong"
    if reference_similarity < 0.7:
        return "moderate"
    if reference_similarity < 0.9:
        return "light"
    return "near_or_exact"


def evaluate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Prediction evaluation requires at least one row")

    source_words = prediction_words = overlap_words = 0
    source_jaccards: list[float] = []
    target_jaccards: list[float] = []
    exact_copies = near_copies = 0
    numeral_rows = numeral_preserved = 0
    negation_agreements = 0
    buckets: defaultdict[str, list[dict[str, float]]] = defaultdict(list)

    for row in rows:
        source = normalized_words(str(row["neutral_text"]))
        target = normalized_words(str(row["target_text"]))
        prediction = normalized_words(str(row["generated_text"]))
        overlap = _overlap_count(source, prediction)
        source_similarity = multiset_jaccard(source, prediction)
        target_similarity = multiset_jaccard(target, prediction)
        reference_similarity = multiset_jaccard(source, target)

        source_words += len(source)
        prediction_words += len(prediction)
        overlap_words += overlap
        source_jaccards.append(source_similarity)
        target_jaccards.append(target_similarity)
        exact_copies += source == prediction
        near_copies += source != prediction and source_similarity >= 0.9

        source_numbers = normalized_numbers(str(row["neutral_text"]))
        if source_numbers:
            numeral_rows += 1
            prediction_numbers = normalized_numbers(str(row["generated_text"]))
            numeral_preserved += not (source_numbers - prediction_numbers)
        negation_agreements += _has_negation(source) == _has_negation(prediction)
        buckets[_transformation_bucket(reference_similarity)].append(
            {
                "source_jaccard": source_similarity,
                "target_jaccard": target_similarity,
                "exact_copy": float(source == prediction),
            }
        )

    count = len(rows)
    by_strength = {}
    for name in ("strong", "moderate", "light", "near_or_exact"):
        entries = buckets[name]
        by_strength[name] = {
            "rows": len(entries),
            "source_jaccard": (
                sum(entry["source_jaccard"] for entry in entries) / len(entries)
                if entries
                else None
            ),
            "target_jaccard": (
                sum(entry["target_jaccard"] for entry in entries) / len(entries)
                if entries
                else None
            ),
            "exact_source_copy_rate": (
                sum(entry["exact_copy"] for entry in entries) / len(entries)
                if entries
                else None
            ),
        }

    return {
        "rows": count,
        "prediction_words_from_source": overlap_words / max(1, prediction_words),
        "source_words_retained": overlap_words / max(1, source_words),
        "source_jaccard": sum(source_jaccards) / count,
        "target_jaccard": sum(target_jaccards) / count,
        "exact_source_copy_rate": exact_copies / count,
        "near_source_copy_rate": near_copies / count,
        "source_numeral_rows": numeral_rows,
        "source_numeral_retention_rate": (
            numeral_preserved / numeral_rows if numeral_rows else None
        ),
        "source_negation_agreement_rate": negation_agreements / count,
        "by_reference_transformation_strength": by_strength,
    }


def summarize_prediction_files(
    named_paths: Mapping[str, str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Score several prediction files on the shared test set and render one table."""
    if not named_paths:
        raise ValueError("Summary requires at least one prediction file")
    summaries: dict[str, dict[str, Any]] = {}
    expected_ids: set[str] | None = None
    for name, path in named_paths.items():
        rows = list(read_jsonl(path))
        row_ids = {str(row["pair_id"]) for row in rows}
        if expected_ids is None:
            expected_ids = row_ids
        elif row_ids != expected_ids:
            raise ValueError(f"Prediction file {name!r} covers different pair IDs")
        summaries[name] = evaluate_prediction_rows(rows)

    header = ["metric", *summaries]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for metric in KEY_METRICS:
        cells = [metric]
        for name in summaries:
            value = summaries[name][metric]
            cells.append("n/a" if value is None else f"{value:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    report = {"models": summaries, "markdown_table": "\n".join(lines)}
    write_json(output_path, report)
    return report


def create_predictions_review(
    named_paths: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    sample_size: int = 50,
) -> dict[str, Any]:
    """Render a private side-by-side sample of every model's outputs for human review."""
    import hashlib

    models = {
        name: {str(row["pair_id"]): row for row in read_jsonl(path)}
        for name, path in named_paths.items()
    }
    first = next(iter(models.values()))
    ordered = sorted(first, key=lambda pid: hashlib.sha256(pid.encode()).hexdigest())
    chosen = ordered[:sample_size]
    lines = ["# Rewrite model review sample", ""]
    for pair_id in chosen:
        row = first[pair_id]
        lines.append(f"## {pair_id}")
        lines.append(f"**Draft:** {row['neutral_text']}")
        lines.append(f"**Gold:** {row['target_text']}")
        for name, rows in models.items():
            lines.append(f"- **{name}:** {rows[pair_id]['generated_text']}")
        lines.append("")
    path = Path(output_path)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o600)
    return {"sampled": len(chosen), "models": list(models), "private_review": str(path)}


def compare_prediction_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    baseline_rows = list(read_jsonl(baseline_path))
    candidate_rows = list(read_jsonl(candidate_path))
    baseline_ids = {str(row["pair_id"]) for row in baseline_rows}
    candidate_ids = {str(row["pair_id"]) for row in candidate_rows}
    if baseline_ids != candidate_ids:
        raise ValueError("Baseline and candidate predictions must contain identical pair IDs")

    baseline = evaluate_prediction_rows(baseline_rows)
    candidate = evaluate_prediction_rows(candidate_rows)
    report = {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": {
            key: (
                candidate[key] - baseline[key]
                if candidate[key] is not None and baseline[key] is not None
                else None
            )
            for key in KEY_METRICS
        },
    }
    write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PREDICTIONS")
    parser.add_argument("--review")
    parser.add_argument("--review-samples", type=int, default=50)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.model:
        named = dict(entry.split("=", maxsplit=1) for entry in arguments.model)
        report = summarize_prediction_files(named, arguments.output)
        print(report["markdown_table"])
        if arguments.review:
            print(
                create_predictions_review(
                    named,
                    arguments.review,
                    sample_size=arguments.review_samples,
                )
            )
        return
    if not (arguments.baseline and arguments.candidate):
        parser.error("Provide either --model entries or both --baseline and --candidate")
    print(compare_prediction_files(arguments.baseline, arguments.candidate, arguments.output))


if __name__ == "__main__":
    main()
