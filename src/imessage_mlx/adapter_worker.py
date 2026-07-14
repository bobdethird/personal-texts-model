from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from imessage_mlx.config import load_yaml
from imessage_mlx.progress import ProgressBar
from imessage_mlx.utils import ensure_private_dir, read_jsonl, sha256_file, write_json, write_jsonl


def _chmod_private_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


def _device(torch):
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _qwen_model_reference(config: dict[str, Any]) -> str:
    from huggingface_hub import snapshot_download

    model_id = str(config["base_model"])
    revision = str(config["revision"])
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    snapshot = hub / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if snapshot.exists():
        return str(snapshot)
    return snapshot_download(model_id, revision=revision)


def train_bart(
    config_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    initial_adapter_dir: str | Path | None = None,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    device = _device(torch)
    output = ensure_private_dir(output_dir)
    train_rows = list(read_jsonl(Path(data_dir) / "train.jsonl"))
    valid_rows = list(read_jsonl(Path(data_dir) / "valid.jsonl"))
    if not train_rows or not valid_rows:
        raise ValueError("BART adapter training requires non-empty train and valid JSONL")

    base_model = str(config["base_model"])
    revision = str(config.get("revision", "main"))
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    max_length = int(config.get("max_length", 256))

    def within_token_limit(row: dict[str, Any]) -> bool:
        source_ids = tokenizer(
            str(row["source"]),
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        target_ids = tokenizer(
            str(row["target"]),
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        return len(source_ids) <= max_length and len(target_ids) <= max_length

    original_train_count = len(train_rows)
    original_valid_count = len(valid_rows)
    train_rows = [row for row in train_rows if within_token_limit(row)]
    valid_rows = [row for row in valid_rows if within_token_limit(row)]
    if not train_rows or not valid_rows:
        raise ValueError("BART length filtering removed every train or validation example")
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, revision=revision)
    initial_adapter_path = Path(initial_adapter_dir).resolve() if initial_adapter_dir else None
    initial_adapter_hash = None
    if initial_adapter_path is not None:
        adapter_path = (
            initial_adapter_path / "adapter"
            if (initial_adapter_path / "adapter").is_dir()
            else initial_adapter_path
        )
        adapter_weights = adapter_path / "adapter_model.safetensors"
        if not adapter_weights.exists():
            raise FileNotFoundError(f"Initial BART adapter weights are missing: {adapter_weights}")
        initial_adapter_hash = sha256_file(adapter_weights)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=int(config.get("lora_rank", 8)),
                lora_alpha=int(config.get("lora_alpha", 16)),
                lora_dropout=float(config.get("lora_dropout", 0.1)),
                target_modules=["q_proj", "v_proj"],
            ),
        )
    model.to(device)

    class PairDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            row = self.rows[index]
            return tokenizer(
                str(row["source"]),
                text_target=str(row["target"]),
                max_length=max_length,
                truncation=True,
            )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        PairDataset(train_rows),
        batch_size=int(config.get("batch_size", 2)),
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
    )
    valid_loader = DataLoader(
        PairDataset(valid_rows),
        batch_size=int(config.get("batch_size", 2)),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    accumulation = int(config.get("gradient_accumulation_steps", 8))
    epochs = int(config.get("epochs", 3))
    total_updates = max(1, math.ceil(len(train_loader) / accumulation) * epochs)
    warmup_updates = int(total_updates * float(config.get("warmup_fraction", 0.05)))

    def learning_rate(update: int) -> float:
        if update < warmup_updates:
            return max(1, update + 1) / max(1, warmup_updates)
        progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    patience = int(config.get("early_stopping_patience", 2))
    best_loss = math.inf
    stale_epochs = 0
    updates = 0
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_progress = ProgressBar(f"Epoch {epoch + 1}/{epochs} train", len(train_loader))
        try:
            for batch_index, batch in enumerate(train_loader):
                batch = {key: value.to(device) for key, value in batch.items()}
                loss = model(**batch).loss / accumulation
                loss.backward()
                train_loss += float(loss.item()) * accumulation
                if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(
                    train_loader
                ):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                train_progress.advance()
        finally:
            train_progress.close()

        model.eval()
        validation_loss = 0.0
        validation_batches = 0
        validation_progress = ProgressBar(
            f"Epoch {epoch + 1}/{epochs} valid",
            len(valid_loader),
        )
        with torch.no_grad():
            try:
                for batch in valid_loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    validation_loss += float(model(**batch).loss.item())
                    validation_batches += 1
                    validation_progress.advance()
            finally:
                validation_progress.close()
        validation_loss /= max(1, validation_batches)
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": train_loss / max(1, len(train_loader)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale_epochs = 0
            model.save_pretrained(output / "adapter", safe_serialization=True)
            tokenizer.save_pretrained(output / "tokenizer")
            _chmod_private_tree(output)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    elapsed = time.perf_counter() - started
    peak_memory = int(torch.mps.driver_allocated_memory()) if device.type == "mps" else 0
    report = {
        "schema_version": 1,
        "architecture": "bart",
        "base_model": base_model,
        "base_revision": revision,
        "continued_from_adapter": initial_adapter_path is not None,
        "initial_adapter_hash": initial_adapter_hash,
        "train_examples": len(train_rows),
        "validation_examples": len(valid_rows),
        "skipped_oversized_train_examples": original_train_count - len(train_rows),
        "skipped_oversized_validation_examples": original_valid_count - len(valid_rows),
        "epochs_completed": len(history),
        "optimizer_updates": updates,
        "best_validation_loss": best_loss,
        "elapsed_seconds": elapsed,
        "examples_per_second": len(train_rows) * len(history) / max(elapsed, 1e-9),
        "peak_memory_bytes": peak_memory,
        "device": str(device),
        "history": history,
        "private_local_artifact": True,
    }
    write_json(output / "training-report.json", report)
    _chmod_private_tree(output)
    return report


def predict_bart(
    config_path: str | Path,
    data_path: str | Path,
    adapter_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    config = load_yaml(config_path)
    device = _device(torch)
    base_model = str(config["base_model"])
    revision = str(config.get("revision", "main"))
    tokenizer_path = Path(adapter_dir) / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, revision=revision)
    model = PeftModel.from_pretrained(model, Path(adapter_dir) / "adapter")
    model.to(device)
    model.eval()
    rows = list(read_jsonl(data_path))
    started = time.perf_counter()

    def predictions():
        with torch.no_grad():
            for row in rows:
                encoded = tokenizer(
                    str(row["source"]),
                    return_tensors="pt",
                    max_length=int(config.get("max_length", 256)),
                    truncation=True,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=int(config.get("max_new_tokens", 64)),
                )
                metadata = {
                    key: str(row[key])
                    for key in ("target_id", "variant_kind", "split", "source_pair_id")
                    if key in row
                }
                yield {
                    "pair_id": str(row["pair_id"]),
                    "neutral_text": str(row["source"]),
                    "target_text": str(row["target"]),
                    "generated_text": tokenizer.decode(
                        generated[0], skip_special_tokens=True
                    ).strip(),
                    **metadata,
                }

    count = write_jsonl(output_path, predictions())
    elapsed = time.perf_counter() - started
    return {
        "architecture": "bart",
        "predictions": count,
        "elapsed_seconds": elapsed,
        "examples_per_second": count / max(elapsed, 1e-9),
        "output": str(Path(output_path)),
    }


def predict_qwen(
    config_path: str | Path,
    data_path: str | Path,
    adapter_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    from mlx_lm import generate, load

    config = load_yaml(config_path)
    model, tokenizer = load(_qwen_model_reference(config), adapter_path=str(adapter_dir))
    rows = list(read_jsonl(data_path))
    started = time.perf_counter()

    def predictions():
        for row in rows:
            output = generate(
                model,
                tokenizer,
                prompt=str(row["prompt"]),
                max_tokens=int(config.get("max_new_tokens", 64)),
                verbose=False,
            )
            metadata = {
                key: str(row[key])
                for key in ("target_id", "variant_kind", "split", "source_pair_id")
                if key in row
            }
            yield {
                "pair_id": str(row["pair_id"]),
                "neutral_text": str(row["prompt"]).rsplit("Draft:\n", maxsplit=1)[-1],
                "target_text": str(row["completion"]),
                "generated_text": output.strip(),
                **metadata,
            }

    count = write_jsonl(output_path, predictions())
    elapsed = time.perf_counter() - started
    return {
        "architecture": "qwen",
        "predictions": count,
        "elapsed_seconds": elapsed,
        "examples_per_second": count / max(elapsed, 1e-9),
        "output": str(Path(output_path)),
    }


def rewrite_bart(
    config_path: str | Path,
    adapter_dir: str | Path,
    draft: str,
) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    config = load_yaml(config_path)
    device = _device(torch)
    run = Path(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(config["base_model"]),
        revision=str(config.get("revision", "main")),
    )
    model = PeftModel.from_pretrained(model, run / "adapter")
    model.to(device)
    model.eval()
    encoded = tokenizer(
        draft,
        return_tensors="pt",
        max_length=int(config.get("max_length", 256)),
        truncation=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(config.get("max_new_tokens", 64)),
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def rewrite_qwen(
    config_path: str | Path,
    adapter_dir: str | Path,
    draft: str,
) -> str:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    from imessage_mlx.data.adapters import REWRITE_INSTRUCTION

    config = load_yaml(config_path)
    model, tokenizer = load(
        _qwen_model_reference(config),
        adapter_path=str(Path(adapter_dir) / "adapter"),
    )
    return generate(
        model,
        tokenizer,
        prompt=f"{REWRITE_INSTRUCTION}{draft}",
        max_tokens=int(config.get("max_new_tokens", 64)),
        sampler=make_sampler(temp=0.0),
        verbose=False,
    ).strip()


def semantic_evaluation(
    data_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
) -> dict[str, Any]:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    rows = list(read_jsonl(data_path))
    if not rows:
        raise ValueError("Semantic evaluation requires predictions")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_path = snapshot_download(model_name, revision=model_revision)
    model = SentenceTransformer(model_path, device=device)
    neutral = [str(row["neutral_text"]) for row in rows]
    target = [str(row["target_text"]) for row in rows]
    generated = [str(row["generated_text"]) for row in rows]
    neutral_embeddings = model.encode(neutral, normalize_embeddings=True)
    target_embeddings = model.encode(target, normalize_embeddings=True)
    generated_embeddings = model.encode(generated, normalize_embeddings=True)
    generated_source = np.sum(generated_embeddings * neutral_embeddings, axis=1)
    target_source = np.sum(target_embeddings * neutral_embeddings, axis=1)
    generated_target = np.sum(generated_embeddings * target_embeddings, axis=1)
    report = {
        "schema_version": 1,
        "model": model_name,
        "model_revision": model_revision,
        "device": device,
        "examples": len(rows),
        "mean_generated_source_similarity": float(generated_source.mean()),
        "mean_target_source_similarity": float(target_source.mean()),
        "mean_generated_target_similarity": float(generated_target.mean()),
        "generated_source_below_0_80": int((generated_source < 0.80).sum()),
        "generated_source_below_0_85": int((generated_source < 0.85).sum()),
        "target_source_below_0_80": int((target_source < 0.80).sum()),
        "target_source_below_0_85": int((target_source < 0.85).sum()),
        "text_persisted_in_report": False,
        "private_local_evaluation": True,
    }
    write_json(output_path, report)
    return report


def convergence_source_semantics(
    data_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    minimum_similarity: float = 0.25,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
) -> dict[str, Any]:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    if not -1 <= minimum_similarity <= 1:
        raise ValueError("Minimum semantic similarity must be between -1 and 1")
    rows = list(read_jsonl(data_path))
    if not rows:
        raise ValueError("Convergence semantic validation requires generated pairs")
    for row in rows:
        if not isinstance(row.get("source"), str) or not isinstance(row.get("target"), str):
            raise ValueError("Convergence semantic rows require source and target strings")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_path = snapshot_download(model_name, revision=model_revision)
    model = SentenceTransformer(model_path, device=device)
    source_embeddings = model.encode(
        [str(row["source"]) for row in rows],
        normalize_embeddings=True,
    )
    target_embeddings = model.encode(
        [str(row["target"]) for row in rows],
        normalize_embeddings=True,
    )
    similarities = np.sum(source_embeddings * target_embeddings, axis=1)
    validated = [
        {**row, "semantic_similarity": float(score)}
        for row, score in zip(rows, similarities, strict=True)
    ]
    write_jsonl(output_path, validated)
    report = {
        "schema_version": 1,
        "task": "convergence_source_semantic_validation",
        "model": model_name,
        "model_revision": model_revision,
        "device": device,
        "examples": len(rows),
        "minimum_required_similarity": minimum_similarity,
        "mean_similarity": float(similarities.mean()),
        "minimum_similarity": float(similarities.min()),
        "below_minimum": int((similarities < minimum_similarity).sum()),
        "accepted_rows": int((similarities >= minimum_similarity).sum()),
        "text_persisted_in_report": False,
        "private_local_evaluation": True,
    }
    write_json(report_path, report)
    return report


def repair_with_qwen(
    config_path: str | Path,
    pairs_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    limit: int,
) -> dict[str, Any]:
    from mlx_lm import generate, load

    from imessage_mlx.data.repair import repair_low_signal_pairs

    config = load_yaml(config_path)
    model, tokenizer = load(_qwen_model_reference(config))

    def local_generate(stage: str, payload: str) -> str:
        messages = [{"role": "user", "content": payload}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=384 if stage == "extract" else 128,
            verbose=False,
        ).strip()
        if output.startswith("```"):
            output = output.removeprefix("```json").removeprefix("```")
            output = output.removesuffix("```").strip()
        return output

    report = repair_low_signal_pairs(
        pairs_path,
        output_path,
        report_path,
        generate=local_generate,
        limit=limit,
    )
    report["local_model"] = str(config["base_model"])
    report["local_model_revision"] = str(config["revision"])
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "train-bart",
        "predict-bart",
        "predict-qwen",
        "semantic-eval",
        "convergence-data-semantics",
        "repair-qwen",
        "rewrite-bart",
        "rewrite-qwen",
    ):
        subparser = subparsers.add_parser(command)
        if command.startswith("rewrite-"):
            subparser.add_argument("--config", required=True)
            subparser.add_argument("--adapter", required=True)
            subparser.add_argument("--draft", required=True)
            continue
        subparser.add_argument("--output", required=True)
        if command in {"semantic-eval", "convergence-data-semantics"}:
            subparser.add_argument("--data", required=True)
            if command == "convergence-data-semantics":
                subparser.add_argument("--report", required=True)
                subparser.add_argument("--minimum", type=float, default=0.25)
            continue
        if command == "repair-qwen":
            subparser.add_argument("--config", required=True)
            subparser.add_argument("--data", required=True)
            subparser.add_argument("--report", required=True)
            subparser.add_argument("--limit", required=True, type=int)
            continue
        subparser.add_argument("--config", required=True)
        if command == "train-bart":
            subparser.add_argument("--data", required=True)
            subparser.add_argument("--initial-adapter")
        else:
            subparser.add_argument("--data", required=True)
            subparser.add_argument("--adapter", required=True)
    arguments = parser.parse_args()
    if arguments.command == "rewrite-bart":
        print(rewrite_bart(arguments.config, arguments.adapter, arguments.draft))
        return
    if arguments.command == "rewrite-qwen":
        print(rewrite_qwen(arguments.config, arguments.adapter, arguments.draft))
        return
    if arguments.command == "semantic-eval":
        report = semantic_evaluation(arguments.data, arguments.output)
    elif arguments.command == "convergence-data-semantics":
        report = convergence_source_semantics(
            arguments.data,
            arguments.output,
            arguments.report,
            minimum_similarity=arguments.minimum,
        )
    elif arguments.command == "repair-qwen":
        report = repair_with_qwen(
            arguments.config,
            arguments.data,
            arguments.output,
            arguments.report,
            limit=arguments.limit,
        )
    elif arguments.command == "train-bart":
        report = train_bart(
            arguments.config,
            arguments.data,
            arguments.output,
            initial_adapter_dir=arguments.initial_adapter,
        )
    elif arguments.command == "predict-bart":
        report = predict_bart(arguments.config, arguments.data, arguments.adapter, arguments.output)
    else:
        report = predict_qwen(arguments.config, arguments.data, arguments.adapter, arguments.output)
    print(report)


if __name__ == "__main__":
    main()
