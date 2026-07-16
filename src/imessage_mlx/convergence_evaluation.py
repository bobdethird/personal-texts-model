from __future__ import annotations

import hashlib
import html
import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean, pvariance
from typing import Any

from imessage_mlx.data.adapters import normalized_fingerprint, protected_facts
from imessage_mlx.rewrite_evaluation import (
    STRUCTURAL_RE,
    _has_repetition,
    _similarity,
    _style_distance,
    _style_markers,
    _word_ngrams,
)
from imessage_mlx.utils import atomic_write_text, read_jsonl, write_json

STYLE_KINDS = (
    "formal_professional",
    "neutral_everyday",
    "verbose_indirect",
    "terse_conversational",
)
_STYLE_SET = frozenset(STYLE_KINDS)
_MARKER_NAMES = tuple(_style_markers([""]).keys())
_TRAIN_TARGET_KEYS = ("target_text", "target", "styled_text", "completion")

JsonInput = str | Path | Mapping[str, Any]


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _aliased_text(
    record: Mapping[str, Any],
    primary: str,
    fallback: str,
    *,
    row_number: int,
) -> str:
    primary_value = record.get(primary)
    fallback_value = record.get(fallback)
    if primary in record and fallback in record and primary_value != fallback_value:
        raise ValueError(
            f"Prediction row {row_number} has conflicting {primary!r} and {fallback!r}"
        )
    value = primary_value if primary in record else fallback_value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Prediction row {row_number} requires non-empty {primary!r} or {fallback!r}"
        )
    return value


def _validate_predictions(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    if not records:
        raise ValueError("Convergence prediction input contains no records")

    normalized: list[dict[str, str]] = []
    groups: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    targets: dict[str, str] = {}
    pair_ids: set[str] = set()

    for row_number, record in enumerate(records, start=1):
        pair_id = record.get("pair_id")
        target_id = record.get("target_id")
        variant_kind = record.get("variant_kind")
        generated = record.get("generated_text")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"Prediction row {row_number} requires a non-empty pair_id")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError(f"Prediction row {row_number} requires a non-empty target_id")
        if variant_kind not in _STYLE_SET:
            raise ValueError(
                f"Prediction row {row_number} has unknown variant_kind {variant_kind!r}"
            )
        if not isinstance(generated, str):
            raise ValueError(f"Prediction row {row_number} requires string generated_text")

        pair_id = pair_id.strip()
        target_id = target_id.strip()
        if pair_id in pair_ids:
            raise ValueError(f"Duplicate convergence pair_id {pair_id!r}")
        pair_ids.add(pair_id)
        if variant_kind in groups[target_id]:
            raise ValueError(
                f"Target {target_id!r} has duplicate {variant_kind!r} prediction variants"
            )

        source = _aliased_text(
            record,
            "neutral_text",
            "source",
            row_number=row_number,
        )
        target = _aliased_text(
            record,
            "target_text",
            "target",
            row_number=row_number,
        )
        if target_id in targets and targets[target_id] != target:
            raise ValueError(f"Target {target_id!r} has inconsistent target text")
        targets[target_id] = target

        value = {
            "pair_id": pair_id,
            "target_id": target_id,
            "variant_kind": str(variant_kind),
            "source": source,
            "target": target,
            "generated": generated,
        }
        normalized.append(value)
        groups[target_id][str(variant_kind)] = value

    for target_id, variants in groups.items():
        if set(variants) != _STYLE_SET or len(variants) != len(STYLE_KINDS):
            missing = sorted(_STYLE_SET - set(variants))
            unexpected = sorted(set(variants) - _STYLE_SET)
            raise ValueError(
                f"Target {target_id!r} must have exactly four distinct variants; "
                f"missing={missing}, unexpected={unexpected}"
            )

    return normalized, dict(groups)


def _challenge_fingerprint(records: list[dict[str, str]]) -> str:
    descriptor = [
        {
            "pair_id": record["pair_id"],
            "target_id": record["target_id"],
            "variant_kind": record["variant_kind"],
            "source_hash": hashlib.sha256(record["source"].encode()).hexdigest(),
            "target_hash": hashlib.sha256(record["target"].encode()).hexdigest(),
        }
        for record in sorted(
            records,
            key=lambda value: (value["target_id"], value["variant_kind"]),
        )
    ]
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _marker_variance(texts: list[str]) -> tuple[float, dict[str, float]]:
    markers = [_style_markers([text]) for text in texts]
    by_marker: dict[str, float] = {}
    for name in _MARKER_NAMES:
        values = [marker[name] for marker in markers]
        variance = pvariance(values)
        if name == "average_characters":
            variance /= max(1.0, _mean(values) ** 2)
        by_marker[name] = variance
    return _mean(list(by_marker.values())), by_marker


def _style_gap_closed(input_distance: float, output_distance: float) -> float:
    if input_distance == 0:
        return 1.0 if output_distance == 0 else 0.0
    return 1.0 - output_distance / input_distance


def _load_json(value: JsonInput | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    loaded = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Aggregate JSON evidence must be an object")
    return loaded


def _training_targets(path: str | Path) -> list[str]:
    targets: list[str] = []
    for row_number, record in enumerate(read_jsonl(path), start=1):
        value = next((record[key] for key in _TRAIN_TARGET_KEYS if key in record), None)
        if not isinstance(value, str):
            keys = ", ".join(_TRAIN_TARGET_KEYS)
            raise ValueError(
                f"Training target row {row_number} requires one string field from: {keys}"
            )
        if value.strip():
            targets.append(value)
    return targets


def evaluate_convergence_predictions(
    predictions_path: str | Path,
    train_targets_path: str | Path,
    output_path: str | Path,
    *,
    semantic_report_path: JsonInput | None = None,
    semantic_report: JsonInput | None = None,
) -> dict[str, Any]:
    if semantic_report_path is not None and semantic_report is not None:
        raise ValueError("Pass only one aggregate semantic report")
    records, groups = _validate_predictions(list(read_jsonl(predictions_path)))
    training_targets = _training_targets(train_targets_path)
    known_targets = {normalized_fingerprint(value) for value in training_targets if value.strip()}
    training_8grams = {ngram for value in training_targets for ngram in _word_ngrams(value)}

    per_style_rows: dict[str, list[dict[str, float | bool]]] = {style: [] for style in STYLE_KINDS}
    group_fact_matches: list[bool] = []
    group_agreements: list[float] = []
    input_variances: list[float] = []
    output_variances: list[float] = []
    input_marker_variances: dict[str, list[float]] = {name: [] for name in _MARKER_NAMES}
    output_marker_variances: dict[str, list[float]] = {name: [] for name in _MARKER_NAMES}
    fact_failure_types: Counter[str] = Counter()
    source_target_fact_conflicts = 0
    groups_with_memorized_output = 0

    for target_id in sorted(groups):
        variants = groups[target_id]
        outputs = [variants[style]["generated"] for style in STYLE_KINDS]
        sources = [variants[style]["source"] for style in STYLE_KINDS]
        pair_agreements = [
            _similarity(left, right) for left, right in itertools.combinations(outputs, 2)
        ]
        group_agreement = _mean(pair_agreements)
        group_agreements.append(group_agreement)

        input_variance, input_by_marker = _marker_variance(sources)
        output_variance, output_by_marker = _marker_variance(outputs)
        input_variances.append(input_variance)
        output_variances.append(output_variance)
        for name in _MARKER_NAMES:
            input_marker_variances[name].append(input_by_marker[name])
            output_marker_variances[name].append(output_by_marker[name])

        row_fact_matches: list[bool] = []
        group_memorized = False
        for style in STYLE_KINDS:
            row = variants[style]
            source_facts = protected_facts(row["source"])
            target_facts = protected_facts(row["target"])
            generated_facts = protected_facts(row["generated"])
            source_target_match = source_facts == target_facts
            fact_match = bool(row["generated"].strip()) and target_facts == generated_facts
            if not source_target_match:
                source_target_fact_conflicts += 1
            if not fact_match:
                for name, expected in target_facts.items():
                    if generated_facts[name] != expected:
                        fact_failure_types[name] += 1
            row_fact_matches.append(fact_match)

            source_markers = _style_markers([row["source"]])
            target_markers = _style_markers([row["target"]])
            output_markers = _style_markers([row["generated"]])
            input_distance = _style_distance(source_markers, target_markers)
            output_distance = _style_distance(output_markers, target_markers)
            fingerprint = normalized_fingerprint(row["generated"])
            exact_memorized = bool(fingerprint) and fingerprint in known_targets
            group_memorized = group_memorized or exact_memorized
            sibling_agreement = _mean(
                [
                    _similarity(row["generated"], variants[other]["generated"])
                    for other in STYLE_KINDS
                    if other != style
                ]
            )
            per_style_rows[style].append(
                {
                    "fact_match": fact_match,
                    "input_copy": bool(fingerprint)
                    and fingerprint == normalized_fingerprint(row["source"]),
                    "target_similarity": _similarity(row["generated"], row["target"]),
                    "source_similarity": _similarity(row["generated"], row["source"]),
                    "input_style_distance": input_distance,
                    "output_style_distance": output_distance,
                    "output_style_proximity": 1.0 / (1.0 + output_distance),
                    "style_gap_closed": _style_gap_closed(input_distance, output_distance),
                    "sibling_agreement": sibling_agreement,
                    "empty": not row["generated"].strip(),
                    "repetition": _has_repetition(row["generated"]),
                    "structural": bool(STRUCTURAL_RE.search(row["generated"])),
                    "exact_memorized": exact_memorized,
                    "long_memorized": len(fingerprint) >= 20 and fingerprint in known_targets,
                    "training_8gram": bool(_word_ngrams(row["generated"]) & training_8grams),
                }
            )
        group_fact_matches.append(all(row_fact_matches))
        groups_with_memorized_output += group_memorized

    per_style: dict[str, dict[str, Any]] = {}
    all_rows = [row for style in STYLE_KINDS for row in per_style_rows[style]]
    for style in STYLE_KINDS:
        values = per_style_rows[style]
        per_style[style] = {
            "examples": len(values),
            "protected_fact_preservation_rate": _mean([float(row["fact_match"]) for row in values]),
            "normalized_input_copies": sum(bool(row["input_copy"]) for row in values),
            "normalized_input_copy_rate": _mean([float(row["input_copy"]) for row in values]),
            "mean_target_similarity": _mean([float(row["target_similarity"]) for row in values]),
            "mean_source_similarity": _mean([float(row["source_similarity"]) for row in values]),
            "mean_style_distance_from_target": _mean(
                [float(row["output_style_distance"]) for row in values]
            ),
            "mean_style_proximity_to_target": _mean(
                [float(row["output_style_proximity"]) for row in values]
            ),
            "mean_style_gap_closed": _mean([float(row["style_gap_closed"]) for row in values]),
            "mean_within_target_output_agreement": _mean(
                [float(row["sibling_agreement"]) for row in values]
            ),
            "empty_outputs": sum(bool(row["empty"]) for row in values),
            "repeated_word_outputs": sum(bool(row["repetition"]) for row in values),
            "structural_token_outputs": sum(bool(row["structural"]) for row in values),
        }

    worst_style, worst_metrics = min(
        per_style.items(),
        key=lambda item: (
            float(item[1]["protected_fact_preservation_rate"]),
            float(item[1]["mean_target_similarity"]),
            float(item[1]["mean_within_target_output_agreement"]),
            float(item[1]["mean_style_proximity_to_target"]),
            -float(item[1]["normalized_input_copy_rate"]),
            item[0],
        ),
    )
    input_variance = _mean(input_variances)
    output_variance = _mean(output_variances)
    marker_reduction = input_variance - output_variance
    marker_reduction_rate = (
        1.0
        if input_variance == 0 and output_variance == 0
        else marker_reduction / max(input_variance, 1e-12)
    )
    semantic_evidence = _load_json(
        semantic_report if semantic_report is not None else semantic_report_path
    )
    semantic_mean = _semantic_mean(semantic_evidence)
    semantic_low = _semantic_low_count(semantic_evidence)
    fact_rate = _mean([float(value) for value in group_fact_matches])
    copy_rate = _mean([float(row["input_copy"]) for row in all_rows])
    agreement = _mean(group_agreements)
    promotion_gates = {
        "protected_facts": fact_rate >= 0.98,
        "semantic_evidence": semantic_mean is not None
        and semantic_mean >= 0.90
        and (semantic_low is None or semantic_low <= max(1, math.ceil(len(records) * 0.05))),
        "input_copy_rate": copy_rate <= 0.20,
        "within_target_convergence": agreement >= 0.70,
        "style_gap_closed": _mean([float(row["style_gap_closed"]) for row in all_rows]) > 0,
        "fluency": not any(
            bool(row[key]) for row in all_rows for key in ("empty", "repetition", "structural")
        ),
    }
    report = {
        "schema_version": 1,
        "task": "convergence_evaluation",
        "target_groups": len(groups),
        "examples": len(records),
        "styles": list(STYLE_KINDS),
        "challenge": {
            "target_groups": len(groups),
            "examples": len(records),
            "styles": list(STYLE_KINDS),
            "fingerprint": _challenge_fingerprint(records),
            "complete_groups": True,
        },
        "content": {
            "protected_fact_preservation_rate": fact_rate,
            "protected_fact_preserved_groups": sum(group_fact_matches),
            "protected_fact_failed_groups": len(group_fact_matches) - sum(group_fact_matches),
            "row_protected_fact_preservation_rate": _mean(
                [float(row["fact_match"]) for row in all_rows]
            ),
            "protected_fact_failure_types": dict(sorted(fact_failure_types.items())),
            "source_target_fact_conflict_rows": source_target_fact_conflicts,
            "mean_target_similarity": _mean([float(row["target_similarity"]) for row in all_rows]),
            "mean_source_similarity": _mean([float(row["source_similarity"]) for row in all_rows]),
        },
        "copying": {
            "normalized_input_copies": sum(bool(row["input_copy"]) for row in all_rows),
            "normalized_input_copy_rate": copy_rate,
            "by_style": {
                style: {
                    "count": per_style[style]["normalized_input_copies"],
                    "rate": per_style[style]["normalized_input_copy_rate"],
                }
                for style in STYLE_KINDS
            },
        },
        "style": {
            "mean_distance_from_target": _mean(
                [float(row["output_style_distance"]) for row in all_rows]
            ),
            "mean_proximity_to_target": _mean(
                [float(row["output_style_proximity"]) for row in all_rows]
            ),
            "mean_input_distance_from_target": _mean(
                [float(row["input_style_distance"]) for row in all_rows]
            ),
            "mean_gap_closed": _mean([float(row["style_gap_closed"]) for row in all_rows]),
            "input_marker_variance_across_styles": input_variance,
            "output_marker_variance_across_styles": output_variance,
            "marker_variance_reduction": marker_reduction,
            "marker_variance_reduction_rate": marker_reduction_rate,
            "variance_by_marker": {
                name: {
                    "input": _mean(input_marker_variances[name]),
                    "output": _mean(output_marker_variances[name]),
                }
                for name in _MARKER_NAMES
            },
        },
        "convergence": {
            "similarity_helper": "casefolded_sequence_matcher",
            "pairs_per_group": math.comb(len(STYLE_KINDS), 2),
            "mean_within_target_output_pairwise_agreement": agreement,
            "minimum_target_group_pairwise_agreement": min(group_agreements),
        },
        "per_style": per_style,
        "worst_register": {
            "variant_kind": worst_style,
            "selection_rule": (
                "lowest protected-fact rate, then target similarity, output agreement, "
                "style proximity, and highest copy rate"
            ),
            "metrics": worst_metrics,
        },
        "fluency": {
            "empty_outputs": sum(bool(row["empty"]) for row in all_rows),
            "repeated_word_outputs": sum(bool(row["repetition"]) for row in all_rows),
            "structural_token_outputs": sum(bool(row["structural"]) for row in all_rows),
            "by_style": {
                style: {
                    "empty_outputs": per_style[style]["empty_outputs"],
                    "repeated_word_outputs": per_style[style]["repeated_word_outputs"],
                    "structural_token_outputs": per_style[style]["structural_token_outputs"],
                }
                for style in STYLE_KINDS
            },
        },
        "memorization": {
            "exact_training_target_matches": sum(bool(row["exact_memorized"]) for row in all_rows),
            "exact_long_training_target_matches": sum(
                bool(row["long_memorized"]) for row in all_rows
            ),
            "outputs_with_matching_training_8gram": sum(
                bool(row["training_8gram"]) for row in all_rows
            ),
            "target_groups_with_exact_training_target_match": groups_with_memorized_output,
            "training_text_persisted_in_report": False,
        },
        "semantic": semantic_evidence,
        "promotion_gates": promotion_gates,
        "ready_to_promote": all(promotion_gates.values()),
        "message_text_persisted_in_report": False,
    }
    write_json(output_path, report)
    return report


def _report(value: JsonInput) -> dict[str, Any]:
    loaded = _load_json(value)
    assert loaded is not None
    return loaded


def _semantic_mean(value: Mapping[str, Any] | None) -> float | None:
    if value is None:
        return None
    for key in (
        "mean_generated_source_similarity",
        "mean_output_source_similarity",
        "mean_similarity",
    ):
        candidate = value.get(key)
        if isinstance(candidate, int | float) and not isinstance(candidate, bool):
            return float(candidate)
    aggregate = value.get("aggregate")
    return _semantic_mean(aggregate) if isinstance(aggregate, Mapping) else None


def _semantic_low_count(value: Mapping[str, Any] | None) -> int | None:
    if value is None:
        return None
    for key in (
        "generated_source_below_0_80",
        "output_source_below_0_80",
        "below_0_80",
    ):
        candidate = value.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            return candidate
    aggregate = value.get("aggregate")
    return _semantic_low_count(aggregate) if isinstance(aggregate, Mapping) else None


def _comparison_inputs(
    baseline_report: JsonInput | Mapping[str, JsonInput],
    augmented_report: JsonInput | None,
    output_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any], str | Path | None]:
    if (
        isinstance(baseline_report, Mapping)
        and "baseline" in baseline_report
        and "augmented" in baseline_report
        and "task" not in baseline_report
    ):
        reports = baseline_report
        if output_path is None and isinstance(augmented_report, str | Path):
            output_path = augmented_report
        return _report(reports["baseline"]), _report(reports["augmented"]), output_path
    if augmented_report is None:
        raise ValueError("Both baseline and augmented convergence reports are required")
    return _report(baseline_report), _report(augmented_report), output_path


def compare_convergence_evaluations(
    baseline_report: JsonInput | Mapping[str, JsonInput],
    augmented_report: JsonInput | None = None,
    output_path: str | Path | None = None,
    *,
    baseline_semantic_report: JsonInput | None = None,
    augmented_semantic_report: JsonInput | None = None,
    semantic_report: JsonInput | None = None,
    material_semantic_regression: float = 0.02,
    measurable_improvement: float = 1e-9,
) -> dict[str, Any]:
    if material_semantic_regression < 0 or measurable_improvement < 0:
        raise ValueError("Comparison tolerances must be nonnegative")
    baseline, augmented, output_path = _comparison_inputs(
        baseline_report,
        augmented_report,
        output_path,
    )
    baseline_challenge = baseline.get("challenge")
    augmented_challenge = augmented.get("challenge")
    if not isinstance(baseline_challenge, Mapping) or not isinstance(augmented_challenge, Mapping):
        raise ValueError("Convergence reports require challenge metadata")
    challenge_keys = ("fingerprint", "target_groups", "examples", "styles")
    if any(baseline_challenge.get(key) != augmented_challenge.get(key) for key in challenge_keys):
        raise ValueError(
            "Baseline and augmented reports must evaluate the same target/style challenge"
        )

    aggregate_semantic = _load_json(semantic_report)
    baseline_semantic = _load_json(baseline_semantic_report)
    augmented_semantic = _load_json(augmented_semantic_report)
    if aggregate_semantic is not None:
        if baseline_semantic is None and isinstance(aggregate_semantic.get("baseline"), Mapping):
            baseline_semantic = dict(aggregate_semantic["baseline"])
        if augmented_semantic is None and isinstance(aggregate_semantic.get("augmented"), Mapping):
            augmented_semantic = dict(aggregate_semantic["augmented"])
    baseline_semantic = baseline_semantic or (
        dict(baseline["semantic"]) if isinstance(baseline.get("semantic"), Mapping) else None
    )
    augmented_semantic = augmented_semantic or (
        dict(augmented["semantic"]) if isinstance(augmented.get("semantic"), Mapping) else None
    )

    explicit_semantic_result = (
        aggregate_semantic.get("no_material_semantic_regression")
        if aggregate_semantic is not None
        else None
    )
    baseline_semantic_mean = _semantic_mean(baseline_semantic)
    augmented_semantic_mean = _semantic_mean(augmented_semantic)
    baseline_low = _semantic_low_count(baseline_semantic)
    augmented_low = _semantic_low_count(augmented_semantic)
    if isinstance(explicit_semantic_result, bool):
        semantic_passed = explicit_semantic_result
        semantic_evidence_available = True
    else:
        semantic_evidence_available = (
            baseline_semantic_mean is not None and augmented_semantic_mean is not None
        )
        semantic_passed = bool(
            semantic_evidence_available
            and augmented_semantic_mean >= baseline_semantic_mean - material_semantic_regression
            and (
                baseline_low is None
                or augmented_low is None
                or augmented_low
                <= baseline_low + max(1, math.ceil(int(augmented_challenge["examples"]) * 0.01))
            )
        )

    fact_rate = float(augmented["content"]["protected_fact_preservation_rate"])
    baseline_copy = float(baseline["copying"]["normalized_input_copy_rate"])
    augmented_copy = float(augmented["copying"]["normalized_input_copy_rate"])
    style_improvements = {
        "target_style_proximity": (
            float(augmented["style"]["mean_proximity_to_target"])
            > float(baseline["style"]["mean_proximity_to_target"]) + measurable_improvement
        ),
        "within_target_output_agreement": (
            float(augmented["convergence"]["mean_within_target_output_pairwise_agreement"])
            > float(baseline["convergence"]["mean_within_target_output_pairwise_agreement"])
            + measurable_improvement
        ),
        "output_style_marker_variance": (
            float(augmented["style"]["output_marker_variance_across_styles"])
            < float(baseline["style"]["output_marker_variance_across_styles"])
            - measurable_improvement
        ),
    }
    gates = {
        "protected_fact_preservation": {
            "passed": fact_rate >= 0.98,
            "required_minimum": 0.98,
            "augmented": fact_rate,
        },
        "semantic_no_material_regression": {
            "passed": semantic_passed,
            "evidence_available": semantic_evidence_available,
            "maximum_mean_regression": material_semantic_regression,
            "baseline_mean": baseline_semantic_mean,
            "augmented_mean": augmented_semantic_mean,
        },
        "copying_lower_or_equal": {
            "passed": augmented_copy <= baseline_copy,
            "baseline": baseline_copy,
            "augmented": augmented_copy,
        },
        "measurable_style_or_convergence_improvement": {
            "passed": any(style_improvements.values()),
            "minimum_change": measurable_improvement,
            "improvements": style_improvements,
        },
    }
    reasons = [name for name, gate in gates.items() if not bool(gate["passed"])]
    expand_recommended = not reasons
    result = {
        "schema_version": 1,
        "task": "convergence_pilot_comparison",
        "challenge_fingerprint": baseline_challenge["fingerprint"],
        "expand_recommended": expand_recommended,
        "reasons": reasons or ["all_expansion_gates_passed"],
        "gates": gates,
        "weighted_score_used": False,
        "message_text_persisted_in_report": False,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def _evenly_selected(values: list[str], sample_size: int) -> list[str]:
    selected = min(sample_size, len(values))
    if selected == 0:
        return []
    if selected == 1:
        return [values[-1]]
    return [values[round(index * (len(values) - 1) / (selected - 1))] for index in range(selected)]


def create_convergence_comparison_review(
    baseline_predictions: str | Path | Mapping[str, str | Path],
    augmented_predictions: str | Path | None = None,
    output_path: str | Path | None = None,
    *,
    summary_path: str | Path | None = None,
    sample_size: int = 50,
) -> dict[str, Any]:
    if sample_size <= 0:
        raise ValueError("Convergence review sample size must be positive")
    if isinstance(baseline_predictions, Mapping):
        prediction_paths = dict(baseline_predictions)
        if output_path is None and augmented_predictions is not None:
            output_path = augmented_predictions
    else:
        if augmented_predictions is None:
            raise ValueError("An augmented prediction file is required")
        prediction_paths = {
            "baseline": baseline_predictions,
            "augmented": augmented_predictions,
        }
    if len(prediction_paths) != 2:
        raise ValueError("Convergence review requires exactly two model prediction files")
    if output_path is None:
        raise ValueError("Convergence review output path is required")

    loaded: dict[str, tuple[list[dict[str, str]], dict[str, dict[str, dict[str, str]]]]] = {}
    for name, path in prediction_paths.items():
        loaded[str(name)] = _validate_predictions(list(read_jsonl(path)))
    model_names = list(loaded)
    first_records, first_groups = loaded[model_names[0]]
    second_records, second_groups = loaded[model_names[1]]
    if _challenge_fingerprint(first_records) != _challenge_fingerprint(second_records):
        raise ValueError(
            "Review models must contain the same target/style challenge and message text"
        )
    if set(first_groups) != set(second_groups):
        raise ValueError("Review models do not contain the same target groups")

    selected_ids = _evenly_selected(sorted(first_groups), sample_size)
    if not selected_ids:
        raise ValueError("Prediction files contain no shared target groups")
    lines = [
        "# Private Convergence Comparison Review",
        "",
        (
            "Review all four registers as one target group. Reject the group if either model "
            "changes any protected fact."
        ),
        "",
    ]
    for number, target_id in enumerate(selected_ids, start=1):
        lines.extend(
            [
                f"## Target group {number}",
                "",
                f"Target ID: `{html.escape(target_id)}`",
                "",
                "**Reference target**",
                f"<pre>{html.escape(first_groups[target_id][STYLE_KINDS[0]]['target'])}</pre>",
                "",
            ]
        )
        for style in STYLE_KINDS:
            first_row = first_groups[target_id][style]
            lines.extend(
                [
                    f"### {style}",
                    "",
                    "**Source**",
                    f"<pre>{html.escape(first_row['source'])}</pre>",
                    "",
                ]
            )
            for model_name in model_names:
                output = loaded[model_name][1][target_id][style]["generated"]
                lines.extend(
                    [
                        f"**{html.escape(model_name)} output**",
                        f"<pre>{html.escape(output)}</pre>",
                        "",
                    ]
                )
        lines.extend(
            [
                "- [ ] No protected fact changed in any output",
                f"- [ ] {model_names[1]} wins or ties {model_names[0]} for this group",
                "- [ ] Outputs converge toward the same target style across all four sources",
                "- [ ] Reject this group",
                "",
            ]
        )

    review_path = Path(output_path)
    atomic_write_text(review_path, "\n".join(lines) + "\n")
    if summary_path is None:
        summary_path = review_path.with_suffix(".summary.json")
    summary = {
        "schema_version": 1,
        "task": "private_convergence_comparison_review",
        "shared_target_groups": len(first_groups),
        "sampled_target_groups": len(selected_ids),
        "styles": list(STYLE_KINDS),
        "models": model_names,
        "selected_target_ids": selected_ids,
        "private_review": str(review_path),
        "machine_summary": str(Path(summary_path)),
        "human_approval_required": True,
        "review_win_or_tie_threshold": 0.70,
        "human_approved": False,
        "reviewed_target_groups": 0,
        "changed_fact_failures": None,
        "review_instructions": (
            "After reviewing the private Markdown, set human_approved=true, record at least "
            "20 reviewed_target_groups, and set changed_fact_failures=0 only when accurate."
        ),
        "message_text_persisted_in_summary": False,
        "message_text_persisted_only_in_private_review": True,
    }
    write_json(summary_path, summary)
    return summary


def create_convergence_pair_review(
    pairs_path: str | Path,
    output_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    sample_size: int = 20,
) -> dict[str, Any]:
    if sample_size <= 0:
        raise ValueError("Convergence pair review sample size must be positive")
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    targets: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(pairs_path):
        target_id = record.get("target_id")
        style = record.get("variant_kind")
        source = record.get("source")
        target = record.get("target")
        if (
            not isinstance(target_id, str)
            or style not in _STYLE_SET
            or not isinstance(source, str)
            or not isinstance(target, str)
        ):
            raise ValueError("Pair review requires target, source, target_id, and variant_kind")
        if style in groups[target_id]:
            raise ValueError(f"Duplicate review variant {style!r} for target {target_id!r}")
        if target_id in targets and targets[target_id] != target:
            raise ValueError(f"Inconsistent review target text for {target_id!r}")
        targets[target_id] = target
        metadata.setdefault(
            target_id,
            {
                "context_bundle": record.get("context_bundle", {}),
                "semantic": record.get("semantic", {}),
                "grounding_stratum": record.get("grounding_stratum", "unknown"),
                "generation_fingerprint": record.get("generation_fingerprint"),
            },
        )
        groups[target_id][str(style)] = record
    incomplete = [target_id for target_id, values in groups.items() if set(values) != _STYLE_SET]
    if incomplete:
        raise ValueError("Pair review requires complete four-source target groups")
    selected_ids = _evenly_selected(sorted(groups), sample_size)
    if not selected_ids:
        raise ValueError("Pair review input contains no complete target groups")

    lines = [
        "# Private Blind-Pair Review",
        "",
        "Review source meaning against the authored target and conversation-resolved intent.",
        "Reject a group if any source changes a fact, intent, negation, uncertainty, or emotion.",
        "",
    ]
    for number, target_id in enumerate(selected_ids, start=1):
        target_metadata = metadata[target_id]
        context_bundle = target_metadata["context_bundle"]
        semantic = target_metadata["semantic"]
        lines.extend(
            [
                f"## Target group {number}",
                "",
                "**Authored target**",
                f"<pre>{html.escape(targets[target_id])}</pre>",
                "",
                f"Grounding stratum: `{html.escape(str(target_metadata['grounding_stratum']))}`",
                "",
            ]
        )
        if isinstance(context_bundle, dict):
            for title, field in (
                ("Exact reply/thread context", "exact_links"),
                ("Recent conversation turns", "recent_turns"),
                ("Retrieved historical evidence", "retrieved_evidence"),
            ):
                values = context_bundle.get(field, [])
                if not isinstance(values, list) or not values:
                    continue
                lines.extend([f"**{title}**", ""])
                for value in values:
                    if isinstance(value, dict):
                        relation = str(value.get("relation", field))
                        role = str(value.get("role", "unknown"))
                        evidence_id = str(
                            value.get("message_id")
                            or ",".join(str(item) for item in value.get("message_ids", []))
                            or "unidentified"
                        )
                        text = html.escape(str(value.get("text", "")))
                        lines.extend(
                            [
                                f"- `{html.escape(evidence_id)}` / "
                                f"`{html.escape(relation)}` / `{html.escape(role)}`",
                                f"  <pre>{text}</pre>",
                            ]
                        )
                lines.append("")
            glossary = context_bundle.get("glossary_entries", [])
            if isinstance(glossary, list) and glossary:
                lines.extend(["**Approved glossary definitions**", ""])
                for entry in glossary:
                    if isinstance(entry, dict):
                        lines.append(
                            f"- **{html.escape(str(entry.get('term', '')))}**: "
                            f"{html.escape(str(entry.get('definition', '')))}"
                        )
                lines.append("")
        if isinstance(semantic, dict):
            paraphrase = str(semantic.get("resolved_paraphrase", "")).strip()
            if paraphrase:
                lines.extend(
                    [
                        "**Resolved meaning**",
                        f"<pre>{html.escape(paraphrase)}</pre>",
                        "",
                    ]
                )
            propositions = semantic.get("atomic_propositions", [])
            lines.extend(["**Extracted semantic propositions**", ""])
            if isinstance(propositions, list):
                lines.extend(f"- {html.escape(str(proposition))}" for proposition in propositions)
            entities = semantic.get("resolved_entities", [])
            if isinstance(entities, list) and entities:
                lines.extend(["", "**Resolved entities**", ""])
                for entity in entities:
                    if isinstance(entity, dict):
                        evidence = ", ".join(str(value) for value in entity.get("evidence_ids", []))
                        lines.append(
                            f"- **{html.escape(str(entity.get('term', '')))}**: "
                            f"{html.escape(str(entity.get('interpretation', '')))} "
                            f"(evidence: `{html.escape(evidence)}`)"
                        )
            slang = semantic.get("slang_interpretations", [])
            if isinstance(slang, list) and slang:
                lines.extend(["", "**Slang and texting expressions**", ""])
                for item in slang:
                    if isinstance(item, dict):
                        evidence = ", ".join(str(value) for value in item.get("evidence_ids", []))
                        suffix = f" (evidence: `{html.escape(evidence)}`)" if evidence else ""
                        lines.append(
                            f"- **{html.escape(str(item.get('expression', '')))}** — "
                            f"`{html.escape(str(item.get('scope', '')))}`: "
                            f"{html.escape(str(item.get('interpretation', '')))}{suffix}"
                        )
            ambiguities = semantic.get("remaining_ambiguities", [])
            if isinstance(ambiguities, list) and ambiguities:
                lines.extend(["", "**Remaining ambiguities**", ""])
                lines.extend(f"- {html.escape(str(value))}" for value in ambiguities)
            lines.append("")
        for style in STYLE_KINDS:
            record = groups[target_id][style]
            score = record.get("semantic_similarity")
            score_text = (
                f" — local score {float(score):.3f}" if isinstance(score, int | float) else ""
            )
            lines.extend(
                [
                    f"**{style}{score_text}**",
                    f"<pre>{html.escape(str(record['source']))}</pre>",
                    "",
                ]
            )
        lines.extend(
            [
                "- [ ] All four sources preserve the target meaning",
                "- [ ] The resolved meaning states what the target actually communicates",
                "- [ ] Slang and in-group expressions are read and scoped correctly",
                "- [ ] Every interpretation is supported by the displayed evidence",
                "- [ ] Speech act, time, modality, uncertainty, and question intent are unchanged",
                "- [ ] All four sources differ materially from the target wording",
                "- [ ] Reject this group",
                "",
            ]
        )

    review_path = Path(output_path)
    atomic_write_text(review_path, "\n".join(lines) + "\n")
    if summary_path is None:
        summary_path = review_path.with_suffix(".summary.json")
    summary = {
        "schema_version": 1,
        "task": "private_blind_pair_review",
        "complete_target_groups": len(groups),
        "sampled_target_groups": len(selected_ids),
        "selected_target_ids": selected_ids,
        "private_review": str(review_path),
        "human_approved": False,
        "reviewed_target_groups": 0,
        "semantic_or_fact_failures": None,
        "generation_fingerprints": sorted(
            {
                str(value["generation_fingerprint"])
                for value in metadata.values()
                if value.get("generation_fingerprint")
            }
        ),
        "review_instructions": (
            "After reviewing the private Markdown, set human_approved=true, record the number "
            "of reviewed_target_groups, and set semantic_or_fact_failures=0 only if every "
            "approved group is evidence-grounded and meaning-preserving."
        ),
        "message_text_persisted_in_summary": False,
        "message_text_persisted_only_in_private_review": True,
    }
    write_json(summary_path, summary)
    return summary


compare_convergence_pilot = compare_convergence_evaluations
create_convergence_grouped_review = create_convergence_comparison_review
