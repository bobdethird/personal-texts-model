from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from imessage_mlx.utils import read_jsonl, sha256_file, write_json, write_jsonl

REWRITE_INSTRUCTION = (
    "Rewrite the draft in the learned casual texting style. Preserve every fact, intent, "
    "question, negation, and degree of uncertainty. Return only the rewritten message.\n\nDraft:\n"
)
CONVERGENCE_VARIANT_KINDS = (
    "formal_professional",
    "neutral_everyday",
    "verbose_indirect",
    "terse_conversational",
)
PLACEHOLDER_RE = re.compile(r"<\|(?:url|email|phone|attachment)\|>")
NUMBER_RE = re.compile(r"\d+(?:[.:/-]\d+)*")
NEGATION_RE = re.compile(
    r"\b(?:no|not|never|none|nothing|nowhere|neither|nor|cannot|can't|cant|"
    r"don't|dont|doesn't|doesnt|didn't|didnt|won't|wont|wouldn't|wouldnt|"
    r"shouldn't|shouldnt|isn't|isnt|aren't|arent|wasn't|wasnt|weren't|werent)\b",
    re.IGNORECASE,
)


def normalized_fingerprint(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def classify_pair_signal(neutral_text: str, styled_text: str) -> str:
    if neutral_text == styled_text:
        return "exact"
    if normalized_fingerprint(neutral_text) == normalized_fingerprint(styled_text):
        return "surface_only"
    similarity = SequenceMatcher(None, neutral_text.casefold(), styled_text.casefold()).ratio()
    return "high_overlap" if similarity >= 0.95 else "substantive"


def protected_facts(text: str) -> dict[str, object]:
    normalized = text.replace("’", "'").replace("‘", "'")
    return {
        "numbers": tuple(NUMBER_RE.findall(normalized)),
        "placeholders": tuple(PLACEHOLDER_RE.findall(normalized)),
        "negated": bool(NEGATION_RE.search(normalized)),
    }


def protected_facts_match(source: str, candidate: str) -> bool:
    return protected_facts(source) == protected_facts(candidate)


def _validated_pair(record: dict[str, Any]) -> dict[str, Any]:
    required = ("pair_id", "neutral_text", "styled_text")
    if any(
        not isinstance(record.get(key), str) or not str(record[key]).strip() for key in required
    ):
        raise ValueError("Adapter records require non-empty pair_id, neutral_text, and styled_text")
    result = {
        "pair_id": str(record["pair_id"]),
        "neutral_text": str(record["neutral_text"]),
        "styled_text": str(record["styled_text"]),
    }
    for key in ("target_id", "variant_kind", "split", "source_pair_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            result[key] = value
    return result


class _NearDuplicateIndex:
    def __init__(self) -> None:
        self.exact: dict[str, set[str]] = defaultdict(set)
        self.buckets: dict[tuple[str, str, int], list[tuple[str, str]]] = defaultdict(list)

    @staticmethod
    def _keys(value: str) -> list[tuple[str, str, int]]:
        length_band = len(value) // 4
        return [
            (value[:2], value[-2:], band)
            for band in range(max(0, length_band - 1), length_band + 2)
        ]

    def match_kind(self, text: str, *, owner: str | None = None) -> str | None:
        value = normalized_fingerprint(text)
        if value in self.exact and any(existing != owner for existing in self.exact[value]):
            return "normalized"
        if len(value) < 12:
            return None
        candidates = {
            candidate
            for key in self._keys(value)
            for candidate in self.buckets.get(key, ())
            if candidate[1] != owner
        }
        if any(
            SequenceMatcher(None, value, candidate).ratio() >= 0.96
            for candidate, _existing_owner in candidates
        ):
            return "fuzzy"
        return None

    def add(self, text: str, *, owner: str = "") -> None:
        value = normalized_fingerprint(text)
        if not value or owner in self.exact[value]:
            return
        self.exact[value].add(owner)
        if len(value) >= 12:
            key = (value[:2], value[-2:], len(value) // 4)
            self.buckets[key].append((value, owner))


def _sample_evenly(records: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    selected = min(max(0, size), len(records))
    if selected == 0:
        return []
    if selected == 1:
        return [records[-1]]
    return [
        records[round(index * (len(records) - 1) / (selected - 1))] for index in range(selected)
    ]


def _balance_training_signal(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    strata: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("exact", "high_overlap", "substantive", "surface_only")
    }
    for record in records:
        stratum = classify_pair_signal(
            str(record["neutral_text"]),
            str(record["styled_text"]),
        )
        strata[stratum].append(record)
    substantive_count = len(strata["substantive"])
    low_signal_cap = max(1, substantive_count // 4)
    selected_ids = {
        str(record["pair_id"])
        for name, values in strata.items()
        for record in (
            _sample_evenly(values, low_signal_cap) if name in {"exact", "surface_only"} else values
        )
    }
    balanced = [record for record in records if str(record["pair_id"]) in selected_ids]
    counts = Counter(
        classify_pair_signal(
            str(record["neutral_text"]),
            str(record["styled_text"]),
        )
        for record in balanced
    )
    return balanced, {
        name: int(counts.get(name, 0))
        for name in ("exact", "high_overlap", "substantive", "surface_only")
    }


def _bart_record(record: dict[str, Any]) -> dict[str, str]:
    result = {
        "pair_id": str(record["pair_id"]),
        "source": str(record["neutral_text"]),
        "target": str(record["styled_text"]),
    }
    for key in ("target_id", "variant_kind", "split", "source_pair_id"):
        if key in record:
            result[key] = str(record[key])
    return result


def _mlx_record(record: dict[str, Any]) -> dict[str, str]:
    result = {
        "pair_id": str(record["pair_id"]),
        "prompt": f"{REWRITE_INSTRUCTION}{record['neutral_text']}",
        "completion": str(record["styled_text"]),
    }
    for key in ("target_id", "variant_kind", "split", "source_pair_id"):
        if key in record:
            result[key] = str(record[key])
    return result


def _write_formats(root: Path, records_by_split: dict[str, list[dict[str, Any]]]) -> None:
    for name, records in records_by_split.items():
        write_jsonl(root / "bart" / f"{name}.jsonl", (_bart_record(row) for row in records))
        write_jsonl(root / "mlx" / f"{name}.jsonl", (_mlx_record(row) for row in records))


def prepare_adapter_datasets(
    splits_dir: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    benchmark_train_size: int = 2_000,
    benchmark_eval_size: int = 200,
    max_characters: int = 512,
) -> dict[str, Any]:
    source = Path(splits_dir)
    output = Path(output_dir)
    input_names = {"train": "train", "validation": "valid", "test": "test"}
    accepted: dict[str, list[dict[str, Any]]] = {}
    split_report: dict[str, dict[str, int]] = {}
    seen = _NearDuplicateIndex()

    for source_name, output_name in input_names.items():
        records = [_validated_pair(value) for value in read_jsonl(source / f"{source_name}.jsonl")]
        kept: list[dict[str, Any]] = []
        removed = Counter()
        for record in records:
            owner = str(record.get("target_id", record["pair_id"]))
            if (
                max(
                    len(str(record["neutral_text"])),
                    len(str(record["styled_text"])),
                )
                > max_characters
            ):
                removed["length"] += 1
                continue
            if not protected_facts_match(
                str(record["neutral_text"]),
                str(record["styled_text"]),
            ):
                removed["fact_conflict"] += 1
                continue
            matches = [
                seen.match_kind(str(record[key]), owner=owner)
                for key in ("neutral_text", "styled_text")
            ]
            match = next((kind for kind in matches if kind == "normalized"), None)
            match = match or next((kind for kind in matches if kind == "fuzzy"), None)
            if source_name != "train" and match:
                removed[match] += 1
                continue
            kept.append(record)
            seen.add(str(record["neutral_text"]), owner=owner)
            seen.add(str(record["styled_text"]), owner=owner)
        accepted[output_name] = kept
        split_report[source_name] = {
            "input_pairs": len(records),
            "output_pairs": len(kept),
            "removed_cross_split_duplicates": int(removed["normalized"] + removed["fuzzy"]),
            "removed_normalized_duplicates": int(removed["normalized"]),
            "removed_fuzzy_duplicates": int(removed["fuzzy"]),
            "removed_fact_conflicts": int(removed["fact_conflict"]),
            "removed_length_outliers": int(removed["length"]),
        }

    balanced_train, balanced_strata = _balance_training_signal(accepted["train"])
    training_splits = {**accepted, "train": balanced_train}
    split_report["train"]["training_pairs_after_signal_balance"] = len(balanced_train)
    _write_formats(output, training_splits)
    benchmark = {
        "train": _sample_evenly(balanced_train, benchmark_train_size),
        "valid": _sample_evenly(accepted["valid"], benchmark_eval_size),
        "test": _sample_evenly(accepted["test"], benchmark_eval_size),
    }
    _write_formats(output / "benchmark", benchmark)

    strata = Counter(
        classify_pair_signal(str(row["neutral_text"]), str(row["styled_text"]))
        for row in accepted["train"]
    )
    report = {
        "schema_version": 1,
        "task": "rewrite",
        "split_policy": (
            "chronological_then_fact_and_length_filtering_then_normalized_and_fuzzy_deduplication"
        ),
        "maximum_characters": max_characters,
        "splits": split_report,
        "train_strata": {
            name: int(strata.get(name, 0))
            for name in ("exact", "high_overlap", "substantive", "surface_only")
        },
        "balanced_train_strata": balanced_strata,
        "benchmark": {name: len(values) for name, values in benchmark.items()},
        "artifacts": {
            f"{family}/{name}": sha256_file(output / family / f"{name}.jsonl")
            for family in ("bart", "mlx")
            for name in ("train", "valid", "test")
        },
    }
    write_json(report_path, report)
    return report


def _validated_bart_record(record: dict[str, Any], *, artifact: str) -> dict[str, str]:
    required = ("pair_id", "source", "target")
    if any(
        not isinstance(record.get(key), str) or not str(record[key]).strip() for key in required
    ):
        raise ValueError(f"{artifact} records require non-empty pair_id, source, and target")
    return {key: str(value) for key, value in record.items() if isinstance(value, str)}


def _validated_convergence_record(record: dict[str, Any]) -> dict[str, Any]:
    required = (
        "pair_id",
        "target_id",
        "source_pair_id",
        "split",
        "variant_kind",
        "source",
        "target",
    )
    if any(
        not isinstance(record.get(key), str) or not str(record[key]).strip() for key in required
    ):
        raise ValueError(f"Convergence records require non-empty fields: {', '.join(required)}")
    split = str(record["split"])
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unknown convergence split {split!r}")
    variant_kind = str(record["variant_kind"])
    if variant_kind not in CONVERGENCE_VARIANT_KINDS:
        raise ValueError(f"Unknown convergence variant kind {variant_kind!r}")
    result = dict(record)
    score = record.get("semantic_similarity")
    if score is not None and (
        isinstance(score, bool) or not isinstance(score, int | float) or not -1 <= float(score) <= 1
    ):
        raise ValueError("Convergence semantic_similarity must be between -1 and 1")
    return result


def prepare_convergence_adapter_datasets(
    base_bart_dir: str | Path | None,
    convergence_pairs_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    minimum_semantic_similarity: float = 0.70,
    minimum_group_mean_similarity: float = 0.80,
    require_semantic_validation: bool = True,
    include_legacy_base: bool = False,
    review_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    if not -1 <= minimum_semantic_similarity <= 1:
        raise ValueError("Minimum convergence semantic similarity must be between -1 and 1")
    if not -1 <= minimum_group_mean_similarity <= 1:
        raise ValueError("Minimum convergence group mean similarity must be between -1 and 1")

    output = Path(output_dir)
    base: dict[str, list[dict[str, str]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    if include_legacy_base:
        if base_bart_dir is None:
            raise ValueError("Legacy augmentation requires a base BART data directory")
        base_root = Path(base_bart_dir)
        base = {
            split: [
                _validated_bart_record(record, artifact=f"Base BART {split}")
                for record in read_jsonl(base_root / f"{split}.jsonl")
            ]
            for split in ("train", "valid", "test")
        }
    convergence = [
        _validated_convergence_record(record) for record in read_jsonl(convergence_pairs_path)
    ]
    if not convergence:
        raise ValueError("Convergence adapter preparation requires generated pairs")
    fingerprints = {
        str(record["generation_fingerprint"])
        for record in convergence
        if isinstance(record.get("generation_fingerprint"), str)
        and str(record["generation_fingerprint"]).strip()
    }
    fingerprinted_rows = sum(
        isinstance(record.get("generation_fingerprint"), str)
        and bool(str(record["generation_fingerprint"]).strip())
        for record in convergence
    )
    if fingerprints and (len(fingerprints) != 1 or fingerprinted_rows != len(convergence)):
        raise ValueError("Convergence rows must share one complete generation fingerprint")
    context_grounded = any(
        record.get("grounding_stratum") in {"reply", "historical", "glossary"}
        or isinstance(record.get("context_bundle"), dict)
        for record in convergence
    )
    review_summary: dict[str, Any] | None = None
    if context_grounded:
        required_review_groups = len({str(record["target_id"]) for record in convergence})
        if review_summary_path is None:
            raise ValueError("Context-grounded convergence data requires a human review summary")
        review_summary = json.loads(Path(review_summary_path).read_text(encoding="utf-8"))
        if (
            not isinstance(review_summary, dict)
            or review_summary.get("human_approved") is not True
            or int(review_summary.get("reviewed_target_groups", 0)) < required_review_groups
            or review_summary.get("semantic_or_fact_failures") != 0
            or (
                fingerprints
                and set(review_summary.get("generation_fingerprints", [])) != fingerprints
            )
        ):
            raise ValueError("Context-grounded convergence review is incomplete or has failures")

    source_pair_owners: dict[tuple[str, str], str] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_ids: set[str] = set()
    for record in convergence:
        pair_id = str(record["pair_id"])
        if pair_id in pair_ids:
            raise ValueError(f"Duplicate convergence pair_id {pair_id!r}")
        pair_ids.add(pair_id)
        target_id = str(record["target_id"])
        source_key = (str(record["split"]), str(record["source_pair_id"]))
        existing_owner = source_pair_owners.setdefault(source_key, target_id)
        if existing_owner != target_id:
            raise ValueError("A source pair is assigned to multiple convergence targets")
        groups[target_id].append(record)

    accepted_groups: dict[str, list[dict[str, Any]]] = {}
    removed = Counter()
    group_splits: dict[str, str] = {}
    for target_id, records in groups.items():
        splits = {str(record["split"]) for record in records}
        styles = {str(record["variant_kind"]) for record in records}
        targets = {str(record["target"]) for record in records}
        if len(splits) != 1 or len(targets) != 1:
            removed["inconsistent_group"] += 1
            continue
        group_splits[target_id] = next(iter(splits))
        if styles != set(CONVERGENCE_VARIANT_KINDS) or len(records) != len(
            CONVERGENCE_VARIANT_KINDS
        ):
            removed["incomplete_group"] += 1
            continue
        if require_semantic_validation and any(
            record.get("semantic_similarity") is None for record in records
        ):
            removed["missing_semantic_validation"] += 1
            continue
        if any(
            record.get("semantic_similarity") is not None
            and float(record["semantic_similarity"]) < minimum_semantic_similarity
            for record in records
        ):
            removed["semantic_similarity"] += 1
            continue
        scores = [
            float(record["semantic_similarity"])
            for record in records
            if record.get("semantic_similarity") is not None
        ]
        if scores and sum(scores) / len(scores) < minimum_group_mean_similarity:
            removed["semantic_group_mean"] += 1
            continue
        accepted_groups[target_id] = records

    if not accepted_groups:
        raise ValueError("No complete convergence target groups passed preparation gates")

    owner_by_base_pair = {
        source_pair_id: owner
        for (split, source_pair_id), owner in source_pair_owners.items()
        if split == "train"
    }
    seen = _NearDuplicateIndex()
    if include_legacy_base:
        for record in base["train"]:
            owner = owner_by_base_pair.get(record["pair_id"], f"base:{record['pair_id']}")
            seen.add(record["source"], owner=owner)
            seen.add(record["target"], owner=owner)

    accepted_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    for split in ("train", "valid", "test"):
        for target_id in sorted(
            target_id for target_id in accepted_groups if group_splits[target_id] == split
        ):
            records = sorted(
                accepted_groups[target_id],
                key=lambda record: CONVERGENCE_VARIANT_KINDS.index(str(record["variant_kind"])),
            )
            leakage = None
            for record in records:
                leakage = seen.match_kind(str(record["source"]), owner=target_id)
                if leakage:
                    break
                if split != "train":
                    leakage = seen.match_kind(str(record["target"]), owner=target_id)
                    if leakage:
                        break
            if leakage:
                removed[f"cross_split_{leakage}"] += 1
                continue
            accepted_by_split[split].extend(records)
            for record in records:
                seen.add(str(record["source"]), owner=target_id)
                seen.add(str(record["target"]), owner=target_id)

    for split in ("train", "valid", "test"):
        if not accepted_by_split[split]:
            raise ValueError(f"No convergence {split} groups passed leakage and quality gates")

    def with_lineage(record: dict[str, str]) -> dict[str, str]:
        owner = (
            source_pair_owners.get(("train", record["pair_id"]))
            or source_pair_owners.get(("valid", record["pair_id"]))
            or source_pair_owners.get(("test", record["pair_id"]))
        )
        if owner is None:
            return record
        return {
            **record,
            "target_id": owner,
            "variant_kind": "original_neutral",
        }

    generated_train = [
        {
            key: str(record[key])
            for key in (
                "pair_id",
                "source",
                "target",
                "target_id",
                "variant_kind",
                "generation_fingerprint",
            )
            if key in record
        }
        for record in accepted_by_split["train"]
    ]
    generated_valid = [
        {
            key: str(record[key])
            for key in (
                "pair_id",
                "source",
                "target",
                "target_id",
                "variant_kind",
                "generation_fingerprint",
            )
            if key in record
        }
        for record in accepted_by_split["valid"]
    ]
    generated_test = [
        {
            key: str(record[key])
            for key in (
                "pair_id",
                "source",
                "target",
                "target_id",
                "variant_kind",
                "generation_fingerprint",
            )
            if key in record
        }
        for record in accepted_by_split["test"]
    ]
    train = (
        [with_lineage(record) for record in base["train"]] + generated_train
        if include_legacy_base
        else generated_train
    )
    valid = (
        [with_lineage(record) for record in base["valid"]] + generated_valid
        if include_legacy_base
        else generated_valid
    )
    test = (
        [with_lineage(record) for record in base["test"]] + generated_test
        if include_legacy_base
        else generated_test
    )
    bart_output = output / "bart"
    write_jsonl(bart_output / "train.jsonl", train)
    write_jsonl(bart_output / "valid.jsonl", valid)
    write_jsonl(bart_output / "test.jsonl", test)
    for split in ("train", "valid", "test"):
        write_jsonl(
            bart_output / f"challenge-{split}.jsonl",
            (
                {
                    key: str(record[key])
                    for key in (
                        "pair_id",
                        "source",
                        "target",
                        "target_id",
                        "variant_kind",
                        "generation_fingerprint",
                    )
                    if key in record
                }
                for record in accepted_by_split[split]
            ),
        )

    accepted_group_counts = {
        split: len({str(record["target_id"]) for record in accepted_by_split[split]})
        for split in ("train", "valid", "test")
    }
    report = {
        "schema_version": 1,
        "task": "convergence_adapter_data",
        "minimum_semantic_similarity": minimum_semantic_similarity,
        "minimum_group_mean_similarity": minimum_group_mean_similarity,
        "semantic_validation_required": require_semantic_validation,
        "training_mode": (
            "generated_plus_legacy_base" if include_legacy_base else "blind_generated_only"
        ),
        "legacy_base_included": include_legacy_base,
        "generation_fingerprint": next(iter(fingerprints), None),
        "human_review_required": context_grounded,
        "human_review_summary": (
            str(Path(review_summary_path)) if review_summary_path is not None else None
        ),
        "base_rows": {split: len(rows) for split, rows in base.items()},
        "model_rows": {
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
        },
        "input_generated_rows": len(convergence),
        "input_target_groups": len(groups),
        "accepted_target_groups": accepted_group_counts,
        "accepted_generated_rows": {
            split: len(records) for split, records in accepted_by_split.items()
        },
        "removed_target_groups": dict(sorted(removed.items())),
        "effective_target_exposure": {
            "synthetic_rows_per_accepted_target": len(CONVERGENCE_VARIANT_KINDS),
            "all_four_rows_used_in_pilot": True,
        },
        "cross_split_target_group_overlap": 0,
        "artifacts": {
            name: sha256_file(bart_output / f"{name}.jsonl")
            for name in (
                "train",
                "valid",
                "test",
                "challenge-train",
                "challenge-valid",
                "challenge-test",
            )
        },
    }
    write_json(report_path, report)
    return report
