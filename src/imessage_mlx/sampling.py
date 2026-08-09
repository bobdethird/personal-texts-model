from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imessage_mlx.data.pairs import build_context_query
from imessage_mlx.data.sft import SFT_FORMAT
from imessage_mlx.training import (
    IM_END,
    IM_START,
    _render_turn_text,
    configure_tokenizer,
    render_system_prefix,
)
from imessage_mlx.utils import write_json

DEMO_BOUNDARY = "The retrieved examples end here. The current conversation begins now."
DEMO_HEADER = "Example: a past conversation and how the phone's owner replied."
# Demonstrations reuse the recent turns of each retrieved exchange so the owner's
# reply is shown in context; cap the length to keep the prompt affordable.
DEMO_CONTEXT_TURNS = 4


def build_generation_prompt(
    messages: list[dict[str, Any]],
    target_index: int,
) -> tuple[str, str]:
    """Build a prompt up to the assistant header and return (prompt, gold reply)."""
    return build_personalized_prompt(messages, target_index)


def build_personalized_prompt(
    messages: list[dict[str, Any]],
    target_index: int,
    *,
    retrieved_examples: list[dict[str, Any]] | None = None,
    style_card: str | None = None,
) -> tuple[str, str]:
    """Build a next-message prompt with optional retrieved demonstrations and style."""
    if not 0 <= target_index < len(messages):
        raise ValueError(f"target_index {target_index} out of range")
    target = messages[target_index]
    if target.get("role") != "assistant":
        raise ValueError("Generation target must be an assistant turn")

    examples = retrieved_examples or []
    instructions: list[str] = []
    if examples:
        instructions.append(
            "The first conversations, each introduced by an example marker, show how the "
            "phone's owner replied in similar situations. Use them as demonstrations, then "
            "reply only to the final conversation."
        )
    if style_card and style_card.strip():
        instructions.append(f"Personal texting style guide:\n{style_card.strip()}")

    parts = [render_system_prefix("\n\n".join(instructions) or None)]
    for example in examples:
        reply = str(example.get("reply", "")).strip()
        if not reply:
            raise ValueError("Retrieved examples must contain a non-empty reply")
        context_turns = _demo_context_turns(example)
        parts.append(f"{IM_START}system\n{DEMO_HEADER}{IM_END}\n")
        for turn in context_turns:
            parts.append(_render_turn_text(turn))
        parts.append(_render_turn_text({"role": "assistant", "content": reply}))
    if examples:
        parts.append(f"{IM_START}system\n{DEMO_BOUNDARY}{IM_END}\n")
    for message in messages[:target_index]:
        parts.append(_render_turn_text(message))
    parts.append(f"{IM_START}assistant\n")
    return "".join(parts), str(target.get("content", ""))


def _demo_context_turns(example: dict[str, Any]) -> list[dict[str, str]]:
    """Recent turns that precede a retrieved reply, for use as a demonstration."""
    turns: list[dict[str, str]] = []
    for turn in example.get("context_messages") or []:
        role = str(turn.get("role", ""))
        content = str(turn.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            turns.append({"role": role, "content": content})
    if not turns:
        query = str(example.get("query", "")).strip()
        if not query:
            raise ValueError("Retrieved examples must contain context or a query")
        turns = [{"role": "user", "content": query}]
    return turns[-DEMO_CONTEXT_TURNS:]


def _trim_completion(text: str) -> str:
    for marker in (IM_END, IM_START):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def _select_targets(
    records: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
    min_context_turns: int,
    require_preceding_user: bool = False,
) -> list[dict[str, Any]]:
    import random

    if samples < 1:
        raise ValueError("samples must be at least 1")
    if min_context_turns < 0:
        raise ValueError("min_context_turns must be non-negative")

    candidates: list[dict[str, Any]] = []
    for record in records:
        if record.get("format") != SFT_FORMAT:
            continue
        messages = record.get("messages") or []
        supervised = [
            int(index)
            for index in record.get("supervised_indexes", [])
            if 0 <= int(index) < len(messages)
            and messages[int(index)].get("role") == "assistant"
            and int(index) >= min_context_turns
        ]
        for target_index in supervised:
            gold = str(messages[target_index].get("content", "")).strip()
            if not gold or gold.startswith("http") or "<|" in gold:
                continue
            if len(gold) < 2:
                continue
            if require_preceding_user and not any(
                message.get("role") == "user" and str(message.get("content", "")).strip()
                for message in messages[:target_index]
            ):
                continue
            candidates.append(
                {
                    "example_id": record.get("example_id") or record.get("session_id"),
                    "session_id": record.get("session_id") or record.get("example_id"),
                    "target_index": target_index,
                    "messages": messages,
                    "gold": gold,
                    "context_turns": target_index,
                }
            )

    if not candidates:
        raise ValueError("No eligible validation assistant turns found for sampling")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Prefer a mix of short and longer contexts.
    candidates.sort(key=lambda item: item["context_turns"])
    sample_count = min(samples, len(candidates))
    if sample_count == 1:
        return [candidates[len(candidates) // 2]]
    indexes = [
        round(index * (len(candidates) - 1) / (sample_count - 1)) for index in range(sample_count)
    ]
    return [candidates[index] for index in indexes]


def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int | None = None,
) -> str:
    import torch

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    eos_ids = [tokenizer.eos_token_id]
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5) if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion_ids = output[0][encoded["input_ids"].shape[-1] :]
    text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    return _trim_completion(text)


def _format_context(messages: list[dict[str, Any]], target_index: int) -> list[dict[str, str]]:
    preview: list[dict[str, str]] = []
    start = max(0, target_index - 6)
    for message in messages[start:target_index]:
        role = "you" if message.get("role") == "assistant" else "them"
        content = str(message.get("content", "")).strip()
        if len(content) > 280:
            content = content[:277] + "..."
        preview.append({"role": role, "content": content})
    return preview


def _last_user_query(messages: list[dict[str, Any]], target_index: int) -> str:
    for message in reversed(messages[:target_index]):
        if message.get("role") == "user":
            query = str(message.get("content", "")).strip()
            if query:
                return query
    raise ValueError("Generation target has no preceding user message")


def _content_cosines(
    encoder: Any,
    golds: list[str],
    generations: list[str],
) -> list[float]:
    import numpy as np

    from imessage_mlx.retrieval import encode_texts

    gold_embeddings = encode_texts(encoder, golds)
    generated_embeddings = encode_texts(encoder, generations)
    return [float(score) for score in np.sum(gold_embeddings * generated_embeddings, axis=1)]


def run_personalization_sampling(config: dict[str, Any]) -> dict[str, Any]:
    """Compare base, retrieval, and retrieval-plus-style generation on held-out turns."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from imessage_mlx.retrieval import RetrievalIndex, load_embedding_model

    validation_path = Path(config["validation_path"])
    style_card_path = Path(config["style_card_path"])
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing validation data: {validation_path}")
    if not style_card_path.is_file():
        raise FileNotFoundError(f"Missing style card: {style_card_path}")
    style_card = style_card_path.read_text(encoding="utf-8").strip()
    if not style_card:
        raise ValueError("Style card is empty")

    records = [
        json.loads(line)
        for line in validation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    targets = _select_targets(
        records,
        samples=int(config.get("samples", 8)),
        seed=int(config.get("seed", 42)),
        min_context_turns=int(config.get("min_context_turns", 1)),
        require_preceding_user=True,
    )
    retrieval_index = RetrievalIndex.load(config["index_dir"])
    embedding_model = load_embedding_model(retrieval_index.model_name)

    model_name = str(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    configure_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()

    temperature = float(config.get("temperature", 0.7))
    max_new_tokens = int(config.get("max_new_tokens", 128))
    top_k = int(config.get("top_k", 4))
    examples: list[dict[str, Any]] = []
    generation_seed = int(config.get("seed", 42))
    for sample_index, target in enumerate(targets):
        query = _last_user_query(target["messages"], target["target_index"])
        context_query = (
            build_context_query(target["messages"][: target["target_index"]]) or query
        )
        retrieved = retrieval_index.retrieve(
            context_query,
            embedding_model,
            top_k=top_k,
            exclude_session_id=str(target["session_id"]),
        )
        base_prompt, gold = build_generation_prompt(
            target["messages"],
            target["target_index"],
        )
        retrieval_prompt, _ = build_personalized_prompt(
            target["messages"],
            target["target_index"],
            retrieved_examples=retrieved,
        )
        style_prompt, _ = build_personalized_prompt(
            target["messages"],
            target["target_index"],
            retrieved_examples=retrieved,
            style_card=style_card,
        )
        examples.append(
            {
                "example_id": target["example_id"],
                "target_index": target["target_index"],
                "context_turns": target["context_turns"],
                "context": _format_context(
                    target["messages"],
                    target["target_index"],
                ),
                "query": query,
                "retrieval_query": context_query,
                "gold": gold,
                "base": _generate(
                    model,
                    tokenizer,
                    base_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=generation_seed + sample_index,
                ),
                "retrieval": _generate(
                    model,
                    tokenizer,
                    retrieval_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=generation_seed + sample_index,
                ),
                "retrieval_style": _generate(
                    model,
                    tokenizer,
                    style_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=generation_seed + sample_index,
                ),
                "retrieved": retrieved,
            }
        )

    golds = [str(example["gold"]) for example in examples]
    for method in ("base", "retrieval", "retrieval_style"):
        scores = _content_cosines(
            embedding_model,
            golds,
            [str(example[method]) for example in examples],
        )
        for example, score in zip(examples, scores, strict=True):
            example.setdefault("content_cosine", {})[method] = score

    summary = {
        "model_name": model_name,
        "embedding_model": retrieval_index.model_name,
        "validation_path": str(validation_path),
        "index_dir": str(config["index_dir"]),
        "style_card_path": str(style_card_path),
        "samples": len(examples),
        "top_k": top_k,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "metric_note": (
            "content_cosine uses the retrieval encoder and is a content proxy, "
            "not an authorship-verification score"
        ),
        "examples": examples,
    }
    output_path = config.get("output_path")
    if output_path:
        write_json(output_path, summary)
        summary["output_path"] = str(output_path)
    return summary


def run_sampling(config: dict[str, Any]) -> dict[str, Any]:
    """Generate next-message samples from a trained LoRA adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    validation_path = Path(config["validation_path"])
    adapter_dir = Path(config["adapter_dir"])
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing validation data: {validation_path}")
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")

    records = [
        json.loads(line)
        for line in validation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    targets = _select_targets(
        records,
        samples=int(config.get("samples", 8)),
        seed=int(config.get("seed", 42)),
        min_context_turns=int(config.get("min_context_turns", 1)),
    )

    model_name = str(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    configure_tokenizer(tokenizer)

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    base.eval()

    examples: list[dict[str, Any]] = []
    temperature = float(config.get("temperature", 0.7))
    max_new_tokens = int(config.get("max_new_tokens", 128))

    generation_seed = int(config.get("seed", 42))
    for sample_index, target in enumerate(targets):
        prompt, gold = build_generation_prompt(target["messages"], target["target_index"])
        base_reply = _generate(
            base,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=generation_seed + sample_index,
        )
        examples.append(
            {
                "example_id": target["example_id"],
                "target_index": target["target_index"],
                "context_turns": target["context_turns"],
                "context": _format_context(target["messages"], target["target_index"]),
                "gold": gold,
                "base": base_reply,
                "finetuned": None,
            }
        )

    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    for sample_index, (example, target) in enumerate(zip(examples, targets, strict=True)):
        prompt, _ = build_generation_prompt(target["messages"], target["target_index"])
        example["finetuned"] = _generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=generation_seed + sample_index,
        )

    summary = {
        "model_name": model_name,
        "adapter_dir": str(adapter_dir),
        "validation_path": str(validation_path),
        "samples": len(examples),
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "examples": examples,
    }
    output_path = config.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["output_path"] = str(path)
    return summary
