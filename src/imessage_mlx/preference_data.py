"""Build DPO preference pairs from an SFT model's own failures.

For every training draft where the fine-tuned model's output fails a fact- or
style-preservation check (near-copy of the draft, lost numerals, flipped negation,
or length collapse), emit a preference row: the gold target is chosen, the model's
failing output is rejected. Rows where the model already behaves are skipped, so
DPO concentrates on the residual failure modes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from imessage_mlx.adapter_evaluation import NEGATIONS, normalized_numbers
from imessage_mlx.adapter_worker import multiset_jaccard, normalized_words
from imessage_mlx.utils import read_jsonl, write_json, write_jsonl

COPY_JACCARD_THRESHOLD = 0.9
LENGTH_COLLAPSE_RATIO = 0.3
# Short drafts legitimately collapse into initialisms ("What do you mean?" -> "wdym"),
# so the length check only fires on sources long enough to carry droppable content.
LENGTH_COLLAPSE_MIN_SOURCE_WORDS = 5


def prediction_failure_reasons(source: str, prediction: str) -> list[str]:
    source_words = normalized_words(source)
    predicted_words = normalized_words(prediction)
    reasons = []
    if not predicted_words:
        return ["empty"]
    if multiset_jaccard(source_words, predicted_words) >= COPY_JACCARD_THRESHOLD:
        reasons.append("copy")
    if normalized_numbers(source) - normalized_numbers(prediction):
        reasons.append("numeral_loss")
    source_negated = any(word in NEGATIONS for word in source_words)
    if source_negated != any(word in NEGATIONS for word in predicted_words):
        reasons.append("negation_flip")
    if len(source_words) >= LENGTH_COLLAPSE_MIN_SOURCE_WORDS and (
        len(predicted_words) < 2
        or len(predicted_words) / len(source_words) < LENGTH_COLLAPSE_RATIO
    ):
        reasons.append("length_collapse")
    return reasons


def build_preference_pairs(
    train_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    cap: int = 8000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if cap <= 0:
        raise ValueError("Preference cap must be positive")
    gold = {str(row["pair_id"]): row for row in train_rows}
    pairs: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for prediction in predictions:
        pair_id = str(prediction["pair_id"])
        row = gold.get(pair_id)
        if row is None:
            continue
        draft = str(row["source"])
        target = str(row["target"])
        generated = str(prediction["generated_text"])
        if generated.strip() == target.strip():
            continue
        reasons = prediction_failure_reasons(draft, generated)
        if not reasons:
            continue
        for reason in reasons:
            reason_counts[reason] += 1
        pairs.append(
            {
                "pair_id": pair_id,
                "draft": draft,
                "chosen": target,
                "rejected": generated,
                "rejection_reasons": reasons,
            }
        )
        if len(pairs) >= cap:
            break
    report = {
        "predictions_scanned": len(predictions),
        "preference_pairs": len(pairs),
        "cap": cap,
        "failure_reasons": dict(sorted(reason_counts.items())),
    }
    return pairs, report


def write_preference_dataset(
    train_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    cap: int = 8000,
    valid_fraction: float = 0.05,
) -> dict[str, Any]:
    pairs, report = build_preference_pairs(
        list(read_jsonl(train_path)),
        list(read_jsonl(predictions_path)),
        cap=cap,
    )
    if not pairs:
        raise ValueError("No preference pairs were produced; the model passed every check")
    split_at = max(1, int(len(pairs) * (1 - valid_fraction)))
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", pairs[:split_at]), ("valid", pairs[split_at:] or pairs[-1:])):
        destination = output / f"{name}.jsonl"
        write_jsonl(destination, subset)
        destination.chmod(0o600)
    output.chmod(0o700)
    report["splits"] = {"train": split_at, "valid": len(pairs) - split_at or 1}
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--cap", type=int, default=8000)
    arguments = parser.parse_args()
    print(
        write_preference_dataset(
            arguments.train,
            arguments.predictions,
            arguments.output,
            arguments.report,
            cap=arguments.cap,
        )
    )


if __name__ == "__main__":
    main()
