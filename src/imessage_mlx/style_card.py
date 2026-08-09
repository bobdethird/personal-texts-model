from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from imessage_mlx.data.pairs import PAIR_FORMAT
from imessage_mlx.sampling import _generate
from imessage_mlx.training import IM_END, IM_START, _render_turn_text, configure_tokenizer
from imessage_mlx.utils import (
    atomic_write_text,
    read_jsonl,
    sha256_file,
    write_json,
)

STYLE_CARD_PROMPT = """Infer the phone owner's texting style from the examples below.
Write a concise operational style guide for another model that must reply as this person.
Describe observable communication strategies: typical length, directness, tone, casing,
punctuation, emoji use, abbreviations, humor, hedging, follow-up messages, and how replies
change with the incoming message. Do not quote examples, repeat private facts, mention names,
or infer personality traits beyond writing behavior. Use short bullet points."""


def select_style_pairs(
    records: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
    max_source_chars: int,
) -> list[dict[str, Any]]:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if max_source_chars < 1:
        raise ValueError("max_source_chars must be at least 1")

    eligible = [
        record
        for record in records
        if record.get("format") == PAIR_FORMAT
        and record.get("split") == "train"
        and str(record.get("query", "")).strip()
        and str(record.get("reply", "")).strip()
    ]
    if not eligible:
        raise ValueError("No eligible training pairs found for style induction")

    random.Random(seed).shuffle(eligible)
    selected: list[dict[str, Any]] = []
    used_chars = 0
    for record in eligible:
        example_chars = len(str(record["query"])) + len(str(record["reply"]))
        if used_chars + example_chars > max_source_chars:
            continue
        selected.append(record)
        used_chars += example_chars
        if len(selected) >= samples:
            break
    if not selected:
        raise ValueError("max_source_chars is too small for every eligible pair")
    return selected


def build_style_card_prompt(pairs: list[dict[str, Any]]) -> str:
    examples = []
    for index, pair in enumerate(pairs, start=1):
        examples.append(
            f"Example {index}\n"
            f"Incoming: {str(pair['query']).strip()}\n"
            f"Reply: {str(pair['reply']).strip()}"
        )
    user_content = f"{STYLE_CARD_PROMPT}\n\n" + "\n\n".join(examples)
    return (
        f"{IM_START}system\n"
        "You analyze writing patterns while excluding topics, identities, and private facts."
        f"{IM_END}\n"
        + _render_turn_text(
            {
                "role": "user",
                "content": user_content,
            }
        )
        + f"{IM_START}assistant\n"
    )


def run_style_card(config: dict[str, Any]) -> dict[str, Any]:
    """Induce and privately cache a style card with one base-model generation."""
    pairs_path = Path(config["pairs_path"])
    output_path = Path(config["output_path"])
    report_path = Path(config.get("report_path", output_path.with_suffix(".json")))
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Missing training pairs: {pairs_path}")
    model_name = str(config["model_name"])
    cache_key = {
        "model_name": model_name,
        "pairs_sha256": sha256_file(pairs_path),
        "samples": int(config.get("samples", 128)),
        "seed": int(config.get("seed", 42)),
        "max_source_chars": int(config.get("max_source_chars", 24_000)),
        "max_new_tokens": int(config.get("max_new_tokens", 768)),
        "temperature": float(config.get("temperature", 0.2)),
    }
    if output_path.is_file() and report_path.is_file() and not bool(config.get("force", False)):
        cached_report = json.loads(report_path.read_text(encoding="utf-8"))
        if all(cached_report.get(key) == value for key, value in cache_key.items()):
            style_card = output_path.read_text(encoding="utf-8").strip()
            return {
                **cached_report,
                "cached": True,
                "style_card": style_card,
            }

    records = list(read_jsonl(pairs_path))
    selected = select_style_pairs(
        records,
        samples=int(config.get("samples", 128)),
        seed=int(config.get("seed", 42)),
        max_source_chars=int(config.get("max_source_chars", 24_000)),
    )
    prompt = build_style_card_prompt(selected)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Style induction requires the personalization dependencies; "
            "run `uv sync --extra personalize`."
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    configure_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    style_card = _generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=int(config.get("max_new_tokens", 768)),
        temperature=float(config.get("temperature", 0.2)),
        seed=int(config.get("seed", 42)),
    )
    if not style_card:
        raise ValueError("The model generated an empty style card")

    atomic_write_text(output_path, style_card.strip() + "\n")
    report = {
        **cache_key,
        "pairs_path": str(pairs_path),
        "selected_pairs": len(selected),
        "selected_pair_ids": [str(pair["pair_id"]) for pair in selected],
        "output_path": str(output_path),
    }
    write_json(report_path, report)
    return {**report, "cached": False, "style_card": style_card}
