"""Lightweight evaluation for base, SFT, and CPO rewrite candidates."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from imessage_mlx.stamp.neutralize import (
    negation_count,
    normalized_numbers,
    placeholder_counts,
)
from imessage_mlx.stamp.rewards import CandidateReward, RewardExponents
from imessage_mlx.utils import write_json

EVALUATION_FORMAT = "stamp-evaluation-v1"
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")


def normalized_words(text: str) -> list[str]:
    return [word.lower().replace("’", "'") for word in _WORD_RE.findall(text)]


def multiset_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_counts, right_counts = Counter(left), Counter(right)
    union = sum((left_counts | right_counts).values())
    if not union:
        return 1.0
    return sum((left_counts & right_counts).values()) / union


def token_fscore(hypothesis: Sequence[str], reference: Sequence[str]) -> float:
    if not hypothesis and not reference:
        return 1.0
    if not hypothesis or not reference:
        return 0.0
    overlap = sum((Counter(hypothesis) & Counter(reference)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(hypothesis)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def character_fscore(
    hypothesis: str,
    reference: str,
    *,
    max_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Case-sensitive whitespace-stripped chrF for target-style similarity."""

    hypothesis_chars = "".join(hypothesis.split())
    reference_chars = "".join(reference.split())
    if not hypothesis_chars and not reference_chars:
        return 1.0
    squared_beta = beta * beta
    scores: list[float] = []
    for order in range(1, max_order + 1):
        hypothesis_grams = Counter(
            hypothesis_chars[index : index + order]
            for index in range(len(hypothesis_chars) - order + 1)
        )
        reference_grams = Counter(
            reference_chars[index : index + order]
            for index in range(len(reference_chars) - order + 1)
        )
        hypothesis_total = sum(hypothesis_grams.values())
        reference_total = sum(reference_grams.values())
        if not hypothesis_total and not reference_total:
            continue
        if not hypothesis_total or not reference_total:
            scores.append(0.0)
            continue
        overlap = sum((hypothesis_grams & reference_grams).values())
        precision = overlap / hypothesis_total
        recall = overlap / reference_total
        if precision + recall == 0:
            scores.append(0.0)
            continue
        scores.append(
            (1 + squared_beta)
            * precision
            * recall
            / (squared_beta * precision + recall)
        )
    return sum(scores) / len(scores) if scores else 0.0


def novel_word_rate(text_words: Sequence[str], source_words: Sequence[str]) -> float:
    if not text_words:
        return 0.0
    source = set(source_words)
    return sum(word not in source for word in text_words) / len(text_words)


def shared_order_agreement(
    left_words: Sequence[str],
    right_words: Sequence[str],
) -> float | None:
    shared = [
        word
        for word, count in (Counter(left_words) & Counter(right_words)).items()
        if count == 1
    ]
    if len(shared) < 2:
        return None
    left_positions = {word: left_words.index(word) for word in shared}
    right_positions = {word: right_words.index(word) for word in shared}
    agreements = total = 0
    for index, first in enumerate(shared):
        for second in shared[index + 1 :]:
            total += 1
            agreements += (left_positions[first] < left_positions[second]) == (
                right_positions[first] < right_positions[second]
            )
    return agreements / total


def order_agreement(
    reference_words: Sequence[str],
    source_words: Sequence[str],
    target_words: Sequence[str],
) -> tuple[int, int]:
    """Count source-vs-target order conflicts where a prediction follows source."""

    shared = [
        word
        for word, count in (
            Counter(reference_words) & Counter(source_words) & Counter(target_words)
        ).items()
        if count == 1
    ]
    reference_positions = {word: reference_words.index(word) for word in shared}
    source_positions = {word: source_words.index(word) for word in shared}
    target_positions = {word: target_words.index(word) for word in shared}
    source_matches = conflicts = 0
    for index, first in enumerate(shared):
        for second in shared[index + 1 :]:
            source_order = source_positions[first] < source_positions[second]
            target_order = target_positions[first] < target_positions[second]
            if source_order == target_order:
                continue
            conflicts += 1
            source_matches += (
                reference_positions[first] < reference_positions[second]
            ) == source_order
    return source_matches, conflicts


def structural_correlation(source_text: str, target_text: str) -> float:
    """Blend overlap, opener, and word order into an anchoring score in [0, 1]."""

    source = normalized_words(source_text)
    target = normalized_words(target_text)
    if not source or not target:
        return 0.0
    order = shared_order_agreement(source, target)
    return (
        0.4 * multiset_jaccard(source, target)
        + 0.2 * (source[0] == target[0])
        + 0.4 * (1.0 if order is None else order)
    )


def _text(value: Any, *, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return "\n".join(str(item) for item in value)
    raise ValueError(f"{field} must be a string or sequence")


def _row_value(row: Mapping[str, Any], names: Sequence[str], *, field: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f"Evaluation row is missing {field}")


def _bubble_count(value: Any) -> int:
    if isinstance(value, str):
        return value.count("\n") + 1
    if isinstance(value, Sequence):
        return len(value)
    return 0


def _candidate_reward(row: Mapping[str, Any]) -> CandidateReward | None:
    value = row.get("reward")
    if isinstance(value, CandidateReward):
        return value
    required = (
        "style_probability",
        "semantic_similarity",
        "base_model_likelihood",
        "length_ratio",
    )
    if not all(field in row for field in required):
        return None
    return CandidateReward(
        candidate_id=str(row.get("candidate_id", row.get("pair_id", ""))),
        text=_text(
            _row_value(
                row,
                ("generated", "generated_text", "candidate"),
                field="generated candidate",
            ),
            field="generated candidate",
        ),
        style_probability=float(row["style_probability"]),
        semantic_similarity=float(row["semantic_similarity"]),
        base_model_likelihood=float(row["base_model_likelihood"]),
        length_ratio=float(row["length_ratio"]),
    )


def evaluate_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    exponents: RewardExponents | None = None,
) -> dict[str, Any]:
    """Aggregate target quality, content invariants, reward, and anchoring."""

    if not rows:
        raise ValueError("Candidate evaluation requires at least one row")
    weights = exponents or RewardExponents()
    target_chrfs: list[float] = []
    target_f1s: list[float] = []
    target_jaccards: list[float] = []
    source_jaccards: list[float] = []
    novel_rates: list[float] = []
    gold_novel_rates: list[float] = []
    exact_copies = near_copies = 0
    number_agreements = negation_agreements = placeholder_agreements = 0
    bubble_agreements = 0
    opener_disagreements = opener_source = opener_target = 0
    order_conflicts = order_source = 0
    split_disagreements = split_source = split_target = 0
    rewards: list[CandidateReward] = []

    for row in rows:
        source_value = _row_value(
            row,
            ("neutral", "neutral_bubbles", "neutral_text"),
            field="neutral source",
        )
        target_value = _row_value(
            row,
            ("original", "original_bubbles", "target_text"),
            field="original target",
        )
        generated_value = _row_value(
            row,
            ("generated", "generated_text", "candidate"),
            field="generated candidate",
        )
        source_text = _text(source_value, field="neutral source")
        target_text = _text(target_value, field="original target")
        generated_text = _text(generated_value, field="generated candidate")
        source = normalized_words(source_text)
        target = normalized_words(target_text)
        generated = normalized_words(generated_text)

        source_similarity = multiset_jaccard(source, generated)
        target_chrfs.append(character_fscore(generated_text, target_text))
        target_f1s.append(token_fscore(generated, target))
        target_jaccards.append(multiset_jaccard(target, generated))
        source_jaccards.append(source_similarity)
        novel_rates.append(novel_word_rate(generated, source))
        gold_novel_rates.append(novel_word_rate(target, source))
        exact_copies += generated == source
        near_copies += generated != source and source_similarity >= 0.9
        number_agreements += normalized_numbers(source_text) == normalized_numbers(generated_text)
        negation_agreements += negation_count(source_text) == negation_count(generated_text)
        placeholder_agreements += placeholder_counts(source_text) == placeholder_counts(
            generated_text
        )
        bubble_agreements += _bubble_count(source_value) == _bubble_count(generated_value)

        if source and target and generated and source[0] != target[0]:
            opener_disagreements += 1
            opener_source += generated[0] == source[0]
            opener_target += generated[0] == target[0]
        matches, conflicts = order_agreement(generated, source, target)
        order_source += matches
        order_conflicts += conflicts
        source_splits = _bubble_count(source_value)
        target_splits = _bubble_count(target_value)
        generated_splits = _bubble_count(generated_value)
        if source_splits != target_splits:
            split_disagreements += 1
            split_source += generated_splits == source_splits
            split_target += generated_splits == target_splits

        reward = _candidate_reward(row)
        if reward is not None:
            rewards.append(reward)

    count = len(rows)

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    aggregate_reward = (
        sum(reward.aggregate(weights) for reward in rewards) / len(rewards)
        if rewards
        else None
    )
    objective_means = {
        objective: (
            sum(reward.objective_scores()[objective] for reward in rewards) / len(rewards)
            if rewards
            else None
        )
        for objective in ("style", "semantic", "likelihood", "length")
    }
    return {
        "rows": count,
        "target_chrf": mean(target_chrfs),
        "target_f1": mean(target_f1s),
        "target_jaccard": mean(target_jaccards),
        "source_jaccard": mean(source_jaccards),
        "novel_word_rate": mean(novel_rates),
        "gold_novel_word_rate": mean(gold_novel_rates),
        "exact_source_copy_rate": exact_copies / count,
        "near_source_copy_rate": near_copies / count,
        "number_agreement_rate": number_agreements / count,
        "negation_agreement_rate": negation_agreements / count,
        "placeholder_agreement_rate": placeholder_agreements / count,
        "bubble_count_agreement_rate": bubble_agreements / count,
        "reward_rows": len(rewards),
        "mean_aggregate_reward": aggregate_reward,
        "mean_objectives": objective_means,
        "structural_anchoring": {
            "opener_disagreements": opener_disagreements,
            "opener_follows_source": (
                opener_source / opener_disagreements if opener_disagreements else None
            ),
            "opener_follows_target": (
                opener_target / opener_disagreements if opener_disagreements else None
            ),
            "order_conflicts": order_conflicts,
            "order_follows_source": (
                order_source / order_conflicts if order_conflicts else None
            ),
            "split_disagreements": split_disagreements,
            "split_follows_source": (
                split_source / split_disagreements if split_disagreements else None
            ),
            "split_follows_target": (
                split_target / split_disagreements if split_disagreements else None
            ),
        },
    }


def evaluate_model_candidates(
    named_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    exponents: RewardExponents | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare base/SFT/CPO (or other named stages) on identical heldout IDs."""

    if not named_rows:
        raise ValueError("At least one candidate set is required")
    expected_ids: set[str] | None = None
    model_reports: dict[str, Any] = {}
    for name, rows in named_rows.items():
        row_ids = {str(row.get("pair_id", "")) for row in rows}
        if "" in row_ids or len(row_ids) != len(rows):
            raise ValueError(f"Candidate set {name!r} requires unique nonempty pair IDs")
        if expected_ids is None:
            expected_ids = row_ids
        elif row_ids != expected_ids:
            raise ValueError("All candidate sets must cover identical pair IDs")
        model_reports[name] = evaluate_candidate_rows(rows, exponents=exponents)
    report = {
        "format": EVALUATION_FORMAT,
        "reward_exponents": (exponents or RewardExponents()).as_dict(),
        "models": model_reports,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report
