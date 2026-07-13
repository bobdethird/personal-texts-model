from __future__ import annotations

import ast
import html
import json
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from imessage_mlx.data.adapters import (
    classify_pair_signal,
    normalized_fingerprint,
    protected_facts,
    protected_facts_match,
)
from imessage_mlx.utils import atomic_write_text, read_jsonl, write_json, write_jsonl

ABBREVIATION_RE = re.compile(
    r"\b(?:u|ur|rn|tmr|tmrw|pls|plz|idk|imo|tbh|omw|lmk|lol|lmao|bc|cuz|tho|"
    r"wanna|gonna|gotta|ty|thx)\b",
    re.IGNORECASE,
)
CONTRACTION_RE = re.compile(r"\b\w+(?:n't|'m|'re|'ve|'ll|'d|'s)\b", re.IGNORECASE)
STRUCTURAL_RE = re.compile(r"<\|[^|\n]+?\|>")


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def _style_markers(values: list[str]) -> dict[str, float]:
    count = max(1, len(values))
    first_alpha = [
        next((character for character in value if character.isalpha()), "") for value in values
    ]
    return {
        "average_characters": sum(len(value) for value in values) / count,
        "initial_lowercase_rate": sum(character.islower() for character in first_alpha) / count,
        "terminal_punctuation_rate": sum(
            bool(value.rstrip()) and value.rstrip()[-1] in ".!?" for value in values
        )
        / count,
        "contraction_rate": sum(bool(CONTRACTION_RE.search(value)) for value in values) / count,
        "abbreviation_rate": sum(bool(ABBREVIATION_RE.search(value)) for value in values) / count,
        "emoji_rate": sum(
            any(unicodedata.category(character) == "So" for character in value) for value in values
        )
        / count,
    }


def _style_distance(left: dict[str, float], right: dict[str, float]) -> float:
    differences = []
    for key in left:
        scale = max(1.0, right[key]) if key == "average_characters" else 1.0
        differences.append(abs(left[key] - right[key]) / scale)
    return sum(differences) / len(differences)


def _character_ngrams(text: str, size: int = 3) -> list[str]:
    normalized = f" {text.casefold()} "
    return [normalized[index : index + size] for index in range(max(0, len(normalized) - size + 1))]


def _style_classifier(
    train_neutral: list[str],
    train_styled: list[str],
    groups: dict[str, list[str]],
) -> dict[str, float]:
    neutral_counts = Counter(ngram for text in train_neutral for ngram in _character_ngrams(text))
    styled_counts = Counter(ngram for text in train_styled for ngram in _character_ngrams(text))
    vocabulary = set(neutral_counts) | set(styled_counts)
    neutral_total = sum(neutral_counts.values()) + len(vocabulary)
    styled_total = sum(styled_counts.values()) + len(vocabulary)
    log_odds = {
        token: math.log((styled_counts[token] + 1) / styled_total)
        - math.log((neutral_counts[token] + 1) / neutral_total)
        for token in vocabulary
    }

    def probability(text: str) -> float:
        grams = _character_ngrams(text)
        if not grams:
            return 0.5
        score = sum(log_odds.get(token, 0.0) for token in grams) / math.sqrt(len(grams))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))

    return {
        name: sum(probability(text) for text in values) / max(1, len(values))
        for name, values in groups.items()
    }


def _has_repetition(text: str) -> bool:
    words = text.casefold().split()
    return any(words[index : index + 3] == [words[index]] * 3 for index in range(len(words) - 2))


def _word_ngrams(text: str, size: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[index : index + size]) for index in range(max(0, len(words) - size + 1))}


def _transformation_strata(
    neutral: list[str],
    outputs: list[str],
) -> dict[str, int]:
    counts = Counter(
        classify_pair_signal(draft, output) for draft, output in zip(neutral, outputs, strict=True)
    )
    return {
        name: int(counts.get(name, 0))
        for name in ("exact", "surface_only", "high_overlap", "substantive")
    }


def _semantic_gate_passes(semantic: dict[str, Any] | None, examples: int) -> bool:
    if semantic is None:
        return False
    generated_low = int(semantic.get("generated_source_below_0_80", 0))
    target_low = int(semantic.get("target_source_below_0_80", 0))
    return float(semantic["mean_generated_source_similarity"]) >= min(
        0.90, float(semantic["mean_target_source_similarity"]) - 0.02
    ) and generated_low <= target_low + max(1, math.ceil(examples * 0.01))


def predict_legacy_rewrite(
    model_dir: str | Path,
    test_split_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    from imessage_mlx.generate import (
        _load_rewrite_model,
        format_rewrite_prompt,
        generate_ids,
    )

    model, tokenizer = _load_rewrite_model(Path(model_dir))
    rows = list(read_jsonl(test_split_path))
    if limit is not None:
        rows = rows[:limit]
    stop_ids = {
        token_id
        for token_id in (
            tokenizer.token_to_id("<|eos|>"),
            tokenizer.token_to_id("<|turn_end|>"),
        )
        if token_id is not None
    }
    skipped_oversized = 0

    def predictions():
        nonlocal skipped_oversized
        for row in rows:
            neutral_text = str(row.get("neutral_text", row.get("source", "")))
            target_text = str(row.get("styled_text", row.get("target", "")))
            prompt_ids = tokenizer.encode(
                format_rewrite_prompt(neutral_text),
                add_special_tokens=False,
            ).ids
            available = model.config.max_sequence_length - len(prompt_ids)
            if available <= 0:
                skipped_oversized += 1
                continue
            generated = generate_ids(
                model,
                prompt_ids,
                eos_ids=stop_ids,
                max_new_tokens=min(64, available),
                temperature=0.0,
                top_p=0.8,
                repetition_penalty=1.1,
                seed=42,
            )
            yield {
                "pair_id": str(row["pair_id"]),
                "neutral_text": neutral_text,
                "target_text": target_text,
                "generated_text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
            }

    count = write_jsonl(output_path, predictions())
    return {
        "architecture": "legacy_random_initialized_transformer",
        "predictions": count,
        "skipped_oversized": skipped_oversized,
        "output": str(Path(output_path)),
    }


def evaluate_rewrite_predictions(
    predictions_path: str | Path,
    train_split_path: str | Path,
    output_path: str | Path,
    *,
    semantic_report_path: str | Path | None = None,
) -> dict[str, Any]:
    predictions = list(read_jsonl(predictions_path))
    if not predictions:
        raise ValueError("Rewrite prediction input contains no records")

    required = ("pair_id", "neutral_text", "target_text", "generated_text")
    for value in predictions:
        if any(not isinstance(value.get(key), str) for key in required):
            raise ValueError(
                "Predictions require string pair_id, neutral, target, and generated text"
            )

    train = list(read_jsonl(train_split_path))
    train_neutral = [str(value["neutral_text"]) for value in train]
    train_styled = [str(value["styled_text"]) for value in train]
    known_targets = {normalized_fingerprint(value) for value in train_styled if value.strip()}

    neutral = [str(value["neutral_text"]) for value in predictions]
    target = [str(value["target_text"]) for value in predictions]
    generated = [str(value["generated_text"]) for value in predictions]
    target_similarity = [
        _similarity(output, expected) for output, expected in zip(generated, target, strict=True)
    ]
    identity_similarity = [
        _similarity(draft, expected) for draft, expected in zip(neutral, target, strict=True)
    ]
    fact_matches = [
        bool(output.strip()) and protected_facts_match(draft, output)
        for draft, output in zip(neutral, generated, strict=True)
    ]
    fact_failure_types = Counter()
    for draft, output in zip(neutral, generated, strict=True):
        source_facts = protected_facts(draft)
        output_facts = protected_facts(output)
        for key in source_facts:
            if source_facts[key] != output_facts[key]:
                fact_failure_types[key] += 1

    neutral_style = _style_markers(neutral)
    target_style = _style_markers(target)
    generated_style = _style_markers(generated)
    identity_distance = _style_distance(neutral_style, target_style)
    generated_distance = _style_distance(generated_style, target_style)
    style_gap_closed = (
        1.0
        if identity_distance == 0 and generated_distance == 0
        else 1 - generated_distance / max(identity_distance, 1e-9)
    )
    classifier = _style_classifier(
        train_neutral,
        train_styled,
        {"neutral": neutral, "target": target, "generated": generated},
    )
    generated_strata = _transformation_strata(neutral, generated)
    target_strata = _transformation_strata(neutral, target)
    generated_lexical_rate = (
        generated_strata["high_overlap"] + generated_strata["substantive"]
    ) / len(predictions)
    target_lexical_rate = (target_strata["high_overlap"] + target_strata["substantive"]) / len(
        predictions
    )
    lexical_gap_closed = generated_lexical_rate / max(target_lexical_rate, 1e-9)

    empty_outputs = sum(not value.strip() for value in generated)
    structural_outputs = sum(bool(STRUCTURAL_RE.search(value)) for value in generated)
    repeated_outputs = sum(_has_repetition(value) for value in generated)
    memorized = sum(
        bool(value.strip()) and normalized_fingerprint(value) in known_targets
        for value in generated
    )
    memorized_long = sum(
        len(normalized_fingerprint(value)) >= 20 and normalized_fingerprint(value) in known_targets
        for value in generated
    )
    training_8grams = {ngram for value in train_styled for ngram in _word_ngrams(value)}
    generated_8gram_matches = sum(
        bool(_word_ngrams(value) & training_8grams) for value in generated
    )
    fact_rate = sum(fact_matches) / len(fact_matches)
    mean_target_similarity = sum(target_similarity) / len(target_similarity)
    identity_baseline_similarity = sum(identity_similarity) / len(identity_similarity)
    semantic = (
        json.loads(Path(semantic_report_path).read_text(encoding="utf-8"))
        if semantic_report_path is not None
        else None
    )
    semantic_pass = _semantic_gate_passes(semantic, len(predictions))
    ready = (
        fact_rate >= 0.98
        and empty_outputs == 0
        and structural_outputs == 0
        and repeated_outputs == 0
        and mean_target_similarity >= identity_baseline_similarity
        and style_gap_closed >= 0.5
        and lexical_gap_closed >= 0.5
        and semantic_pass
    )

    report = {
        "schema_version": 1,
        "task": "rewrite",
        "examples": len(predictions),
        "content": {
            "mean_target_similarity": mean_target_similarity,
            "identity_baseline_similarity": identity_baseline_similarity,
            "protected_fact_preservation_rate": fact_rate,
            "protected_fact_failures": len(fact_matches) - sum(fact_matches),
            "protected_fact_failure_types": dict(fact_failure_types),
        },
        "style": {
            "neutral": neutral_style,
            "target": target_style,
            "generated": generated_style,
            "distance_from_target": generated_distance,
            "identity_distance_from_target": identity_distance,
            "gap_closed": style_gap_closed,
            "classifier_probability": classifier,
            "generated_transformation_strata": generated_strata,
            "target_transformation_strata": target_strata,
            "generated_lexical_change_rate": generated_lexical_rate,
            "target_lexical_change_rate": target_lexical_rate,
            "lexical_change_gap_closed": lexical_gap_closed,
        },
        "fluency": {
            "empty_outputs": empty_outputs,
            "structural_token_outputs": structural_outputs,
            "repeated_word_outputs": repeated_outputs,
        },
        "memorization": {
            "exact_training_target_matches": memorized,
            "exact_long_training_target_matches": memorized_long,
            "outputs_with_matching_training_8gram": generated_8gram_matches,
            "training_text_persisted_in_report": False,
        },
        "semantic": semantic,
        "ready_to_promote": ready,
    }
    write_json(output_path, report)
    return report


def compare_rewrite_evaluations(
    report_paths: dict[str, str | Path],
    output_path: str | Path,
    *,
    training_report_paths: dict[str, str | Path] | None = None,
    prediction_log_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    if not report_paths:
        raise ValueError("At least one rewrite evaluation report is required")
    candidates: dict[str, dict[str, Any]] = {}
    for name, path in report_paths.items():
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        content = report["content"]
        style = report["style"]
        fluency = report["fluency"]
        semantic = report.get("semantic")
        qualifies = (
            _semantic_gate_passes(semantic, int(report.get("examples", 1)))
            and float(content["protected_fact_preservation_rate"]) >= 0.95
            and float(content["mean_target_similarity"])
            >= float(content["identity_baseline_similarity"])
            and float(style["gap_closed"]) > 0
            and float(style.get("lexical_change_gap_closed", 0.0)) >= 0.5
            and int(fluency["empty_outputs"]) == 0
            and int(fluency["repeated_word_outputs"]) == 0
            and int(fluency["structural_token_outputs"]) == 0
        )
        score = (
            0.5 * float(content["mean_target_similarity"])
            + 0.3 * float(content["protected_fact_preservation_rate"])
            + 0.2 * max(-1.0, min(1.0, float(style["gap_closed"])))
        )
        candidates[name] = {
            "qualifies_for_full_training": qualifies,
            "selection_score": score,
            "content": content,
            "semantic": report.get("semantic"),
            "style_gap_closed": style["gap_closed"],
            "fluency": fluency,
        }
        if training_report_paths and name in training_report_paths:
            training = json.loads(Path(training_report_paths[name]).read_text(encoding="utf-8"))
            log = str(training.get("stdout_tail", ""))
            peak_matches = re.findall(r"Peak mem ([0-9.]+) GB", log)
            token_rate_matches = re.findall(r"Tokens/sec ([0-9.]+)", log)
            efficiency = {
                "training_elapsed_seconds": training.get("elapsed_seconds"),
                "training_examples_per_second": training.get("examples_per_second"),
                "training_tokens_per_second": (
                    float(token_rate_matches[-1]) if token_rate_matches else None
                ),
                "peak_memory_bytes": training.get("peak_memory_bytes"),
                "peak_memory_gb": (float(peak_matches[-1]) if peak_matches else None),
            }
            if prediction_log_paths and name in prediction_log_paths:
                prediction_log = (
                    Path(prediction_log_paths[name]).read_text(encoding="utf-8").splitlines()[0]
                )
                try:
                    prediction = ast.literal_eval(prediction_log)
                except (SyntaxError, ValueError):
                    prediction = {}
                efficiency["prediction_elapsed_seconds"] = prediction.get("elapsed_seconds")
                efficiency["prediction_examples_per_second"] = prediction.get("examples_per_second")
            candidates[name]["efficiency"] = efficiency
    qualified = [
        (name, value) for name, value in candidates.items() if value["qualifies_for_full_training"]
    ]
    selected = (
        max(qualified, key=lambda item: float(item[1]["selection_score"]))[0] if qualified else None
    )
    result = {
        "schema_version": 1,
        "task": "rewrite_architecture_selection",
        "selected": selected,
        "candidates": candidates,
        "selection_requires_content_style_and_fluency": True,
    }
    write_json(output_path, result)
    return result


def create_rewrite_comparison_review(
    prediction_paths: dict[str, str | Path],
    output_path: str | Path,
    *,
    sample_size: int = 50,
) -> dict[str, Any]:
    if sample_size <= 0:
        raise ValueError("Comparison review sample size must be positive")
    by_model = {
        name: {str(row["pair_id"]): row for row in read_jsonl(path)}
        for name, path in prediction_paths.items()
    }
    if not by_model:
        raise ValueError("At least one prediction file is required")
    shared_ids = sorted(set.intersection(*(set(rows) for rows in by_model.values())))
    selected_count = min(sample_size, len(shared_ids))
    if not selected_count:
        raise ValueError("Prediction files contain no shared pair identifiers")
    indices = (
        [len(shared_ids) - 1]
        if selected_count == 1
        else [
            round(index * (len(shared_ids) - 1) / (selected_count - 1))
            for index in range(selected_count)
        ]
    )
    selected_ids = [shared_ids[index] for index in indices]
    first_model = next(iter(by_model))
    lines = [
        "# Private Rewrite Comparison Review",
        "",
        "Review meaning, personal style, and fluency. Do not approve a changed fact.",
        "Record a preference for the neutral draft, legacy model, adapter, or target.",
        "",
    ]
    for number, pair_id in enumerate(selected_ids, start=1):
        source = by_model[first_model][pair_id]
        lines.extend(
            [
                f"## Example {number}",
                "",
                "**Neutral draft**",
                f"<pre>{html.escape(str(source['neutral_text']))}</pre>",
                "",
                "**Reference target**",
                f"<pre>{html.escape(str(source['target_text']))}</pre>",
                "",
            ]
        )
        for name, rows in by_model.items():
            lines.extend(
                [
                    f"**{name}**",
                    f"<pre>{html.escape(str(rows[pair_id]['generated_text']))}</pre>",
                    "",
                ]
            )
        lines.extend(
            [
                "- [ ] Meaning preserved by adapter",
                "- [ ] Adapter sounds more like me than neutral",
                "- [ ] Adapter preferred over legacy",
                "- [ ] Reject: changed fact or malformed",
                "",
            ]
        )
    atomic_write_text(output_path, "\n".join(lines) + "\n")
    report = {
        "schema_version": 1,
        "shared_examples": len(shared_ids),
        "sampled_examples": selected_count,
        "models": list(prediction_paths),
        "private_review": str(Path(output_path)),
        "human_approval_required": True,
        "text_persisted_only_in_private_review": True,
    }
    return report
