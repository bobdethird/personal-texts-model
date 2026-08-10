from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

# Resolve imessage_mlx from src/ even when an editable-install .pth file is hidden
# while Modal packages local source.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

APP_NAME = "imessage-stamp"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ROUNDS = 3
DEFAULT_CANDIDATES = 8
DEFAULT_SEED = 42

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("imessage-sft-data", create_if_missing=True)
artifact_volume = modal.Volume.from_name("imessage-sft-artifacts", create_if_missing=True)
model_cache = modal.Volume.from_name("imessage-sft-model-cache", create_if_missing=True)

stamp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==1.14.0",
        "datasets==5.0.0",
        "numpy==2.4.6",
        "openai==2.53.0",
        "peft==0.19.1",
        "scikit-learn==1.9.0",
        "sentence-transformers==5.7.0",
        "torch==2.13.0",
        "transformers==5.14.1",
        "trl==1.9.2",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "SENTENCE_TRANSFORMERS_HOME": "/cache/sentence-transformers",
            "TOKENIZERS_PARALLELISM": "false",
            # Batched generation allocates variable-length KV caches, which
            # fragments the default allocator badly enough to trigger OOM.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_python_source("imessage_mlx")
)

_VOLUMES = {
    "/data": data_volume,
    "/outputs": artifact_volume,
    "/cache": model_cache,
}
_GPU_FUNCTION_OPTIONS: dict[str, Any] = {
    "image": stamp_image,
    "gpu": "A100-40GB",
    "cpu": 8,
    "memory": 32_768,
    "timeout": 24 * 60 * 60,
    "startup_timeout": 20 * 60,
    "retries": modal.Retries(initial_delay=2.0, max_retries=2),
    "single_use_containers": True,
    "volumes": _VOLUMES,
}
# CPU coordinator so `modal run --detach` can finish later stages after the
# laptop disconnects. Stage work still runs on the GPU functions below.
_ORCHESTRATOR_FUNCTION_OPTIONS: dict[str, Any] = {
    "image": stamp_image,
    "cpu": 2,
    "memory": 4_096,
    "timeout": 24 * 60 * 60,
    "retries": 0,
}

_PREPARE_MODULES = ("imessage_mlx.data.stamp",)
_PREPARE_CALLABLES = (
    "prepare_neutralization_inputs",
    "prepare_stamp_inputs",
    "prepare_stamp_dataset",
    "prepare_stamp_corpus",
    "prepare_style_units",
    "build_style_units",
    "run_preparation",
)
_NEUTRALIZE_MODULES = (
    "imessage_mlx.stamp.neutralize",
    "imessage_mlx.stamp",
)
_NEUTRALIZE_CALLABLES = (
    "run_neutralization",
    "neutralize_dataset",
    "generate_neutral_pairs",
    "neutralize_pairs",
    "run_neutralize",
)
_CLASSIFIER_MODULES = (
    "imessage_mlx.stamp.classifier",
    "imessage_mlx.stamp",
)
_CLASSIFIER_CALLABLES = (
    "train_style_classifier",
    "run_classifier_training",
    "train_binary_classifier",
    "train_classifier",
)
_SFT_MODULES = (
    "imessage_mlx.stamp.sft",
    "imessage_mlx.stamp",
)
_SFT_CALLABLES = (
    "train_initial_sft",
    "train_rewrite_sft",
    "run_sft_training",
    "train_sft",
    "run_initial_sft",
)
_PREFERENCE_MODULES = (
    "imessage_mlx.stamp.preference",
    "imessage_mlx.stamp.rewards",
    "imessage_mlx.stamp",
)
_PREFERENCE_CALLABLES = (
    "generate_and_score_preferences",
    "generate_and_score_candidates",
    "run_preference_round",
    "build_preference_dataset",
    "generate_preferences",
)
_GENERATION_CALLABLES = (
    "generate_preference_candidates",
    "generate_candidates",
    "sample_candidates",
)
_SCORING_CALLABLES = (
    "score_preference_candidates",
    "score_candidates",
    "select_hope_fear_pairs",
    "build_hope_fear_pairs",
)
_CPO_MODULES = (
    "imessage_mlx.stamp.preference",
    "imessage_mlx.stamp.sft",
    "imessage_mlx.stamp",
)
_CPO_CALLABLES = (
    "train_cpo_round",
    "run_cpo_training",
    "train_cpo",
    "run_cpo_round",
)
_EVALUATE_MODULES = (
    "imessage_mlx.stamp.evaluate",
    "imessage_mlx.stamp",
)
_EVALUATE_CALLABLES = (
    "evaluate_stamp",
    "run_evaluation",
    "evaluate_models",
    "evaluate_model_candidates",
    "evaluate",
)

_CHECKPOINT_CALLBACK_NAMES = (
    "on_checkpoint_saved",
    "checkpoint_callback",
    "on_checkpoint",
    "on_save",
    "commit_callback",
)
_PRIVATE_RESULT_KEYS = {
    "candidate",
    "candidates",
    "chosen",
    "completion",
    "completions",
    "content",
    "examples",
    "fear",
    "hope",
    "messages",
    "neutral",
    "original",
    "records",
    "target",
    "text",
}


class StampIntegrationError(RuntimeError):
    """Raised when the orchestration contract and the core package do not match."""


def _safe_run_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-")
    if not name:
        raise ValueError("run_name must contain at least one letter or number")
    return name


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _stamp_roots(run_name: str) -> tuple[str, str]:
    return f"/data/{run_name}/stamp", f"/outputs/{run_name}/stamp"


def _round_name(round_number: int) -> str:
    if round_number < 1:
        raise ValueError("round_number must be at least 1")
    return f"round-{round_number:02d}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _public_summary(value: Any, *, depth: int = 0) -> Any:
    """Keep manifests useful without copying private generated text into them."""
    if depth > 4:
        return "<omitted>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        summary: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key.lower() in _PRIVATE_RESULT_KEYS:
                continue
            summary[normalized_key] = _public_summary(item, depth=depth + 1)
        return summary
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        return [_public_summary(item, depth=depth + 1) for item in items[:20]]
    return repr(value)


def _config_fingerprint(stage: str, config: Mapping[str, Any]) -> str:
    ignored = {"force", "resume"}
    payload = {
        "stage": stage,
        "config": {
            key: _json_safe(value)
            for key, value in sorted(config.items())
            if key not in ignored
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True))
            output.write("\n")
    temporary.replace(destination)


def _read_json(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.is_file():
        return None
    loaded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return loaded


def _read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON objects in {path}")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def _require_file(path: str | Path, description: str) -> None:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {description}: {source}")


def _require_artifacts(path: str | Path, description: str) -> None:
    source = Path(path)
    if source.is_file() and source.stat().st_size > 0:
        return
    if source.is_dir() and any(item.is_file() for item in source.rglob("*")):
        return
    raise FileNotFoundError(f"Missing or empty {description}: {source}")


def _reload_volumes() -> None:
    data_volume.reload()
    artifact_volume.reload()
    model_cache.reload()


def _commit_all() -> None:
    data_volume.commit()
    artifact_volume.commit()
    model_cache.commit()


def _commit_checkpoint(*_args: Any, **_kwargs: Any) -> None:
    # Core trainers can call this after each checkpoint. This bounds loss from a
    # preemption to at most one save interval.
    artifact_volume.commit()
    model_cache.commit()


def _resolve_callable(
    module_names: Sequence[str],
    callable_names: Sequence[str],
    *,
    required: bool = True,
) -> Callable[..., Any] | None:
    import_errors: list[str] = []
    modules = []
    for module_name in module_names:
        try:
            modules.append(importlib.import_module(module_name))
        except ImportError as error:
            import_errors.append(f"{module_name}: {error}")

    for callable_name in callable_names:
        for module in modules:
            candidate = getattr(module, callable_name, None)
            if callable(candidate):
                return candidate

    if not required:
        return None
    searched = ", ".join(
        f"{module_name}.{callable_name}"
        for module_name in module_names
        for callable_name in callable_names
    )
    details = "; ".join(import_errors) or "modules imported but no compatible callable exists"
    raise StampIntegrationError(
        f"Could not resolve a STAMP core callable. Tried {searched}. Details: {details}"
    )


def _invoke_core(
    function: Callable[..., Any],
    config: Mapping[str, Any],
    *,
    checkpointed: bool,
) -> Any:
    """Invoke config-style or keyword-style core APIs without masking core errors."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        kwargs = {"on_checkpoint_saved": _commit_checkpoint} if checkpointed else {}
        return function(dict(config), **kwargs)

    parameters = signature.parameters
    has_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    callback_kwargs: dict[str, Any] = {}
    if checkpointed:
        callback_name = next(
            (name for name in _CHECKPOINT_CALLBACK_NAMES if name in parameters),
            None,
        )
        if callback_name is not None:
            callback_kwargs[callback_name] = _commit_checkpoint
        elif has_var_keyword:
            callback_kwargs["on_checkpoint_saved"] = _commit_checkpoint

    config_parameter = next(
        (
            name
            for name in ("config", "settings", "params", "options")
            if name in parameters
        ),
        None,
    )
    if config_parameter is not None:
        return function(**{config_parameter: dict(config), **callback_kwargs})

    positional_parameters = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name not in callback_kwargs
    ]
    if (
        len(positional_parameters) == 1
        and positional_parameters[0].default is inspect.Parameter.empty
    ):
        return function(dict(config), **callback_kwargs)

    if has_var_keyword:
        return function(**dict(config), **callback_kwargs)

    keyword_config = {
        name: value
        for name, value in config.items()
        if name in parameters
        and parameters[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    missing = [
        parameter.name
        for parameter in parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and parameter.name not in keyword_config
        and parameter.name not in callback_kwargs
    ]
    if missing:
        raise StampIntegrationError(
            f"{function.__module__}.{function.__name__} requires unsupported arguments: "
            f"{', '.join(missing)}"
        )
    return function(**keyword_config, **callback_kwargs)


def _supports_checkpoint_callback(function: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return True
    return any(name in parameters for name in _CHECKPOINT_CALLBACK_NAMES) or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _invoke_with_checkpoint_commits(
    function: Callable[..., Any],
    config: Mapping[str, Any],
    output_dir: str,
) -> Any:
    if _supports_checkpoint_callback(function):
        return _invoke_core(function, config, checkpointed=True)

    stopped = threading.Event()

    def commit_completed_checkpoints() -> None:
        seen: set[tuple[str, int]] = set()
        while not stopped.wait(15):
            completed = {
                (str(path.parent), path.stat().st_mtime_ns)
                for path in Path(output_dir).glob("checkpoint-*/trainer_state.json")
            }
            if completed - seen:
                _commit_checkpoint()
                seen = completed

    watcher = threading.Thread(
        target=commit_completed_checkpoints,
        name="stamp-checkpoint-committer",
        daemon=True,
    )
    watcher.start()
    try:
        return _invoke_core(function, config, checkpointed=False)
    finally:
        stopped.set()
        watcher.join()
        _commit_checkpoint()


def _find_result_path(result: Any, keys: Sequence[str]) -> str | None:
    if not isinstance(result, Mapping):
        return None
    for key in keys:
        candidate = result.get(key)
        if isinstance(candidate, (str, Path)) and str(candidate):
            return str(candidate)
    nested = result.get("result")
    if isinstance(nested, Mapping):
        return _find_result_path(nested, keys)
    return None


def _best_state_path(result: Any, output_dir: str) -> str:
    returned = _find_result_path(
        result,
        (
            "state_path",
            "final_dir",
            "adapter_dir",
            "model_dir",
            "output_dir",
            "checkpoint_dir",
        ),
    )
    if returned and Path(returned).exists():
        return returned
    final_dir = Path(output_dir) / "final"
    if final_dir.exists():
        return str(final_dir)
    return output_dir


def _stage_manifest_path(config: Mapping[str, Any], stage: str) -> str:
    return f"{config['stamp_output_dir']}/manifests/{stage}.json"


def _completed_stage(
    manifest_path: str,
    fingerprint: str,
    expected_output: str,
) -> dict[str, Any] | None:
    manifest = _read_json(manifest_path)
    if (
        manifest
        and manifest.get("status") == "completed"
        and manifest.get("fingerprint") == fingerprint
    ):
        _require_artifacts(expected_output, "resumable stage output")
        return manifest
    return None


def _run_core_stage(
    *,
    stage: str,
    config: dict[str, Any],
    module_names: Sequence[str],
    callable_names: Sequence[str],
    required_inputs: Sequence[tuple[str, str]],
    expected_output: str,
    checkpointed: bool = False,
    runner: Callable[[dict[str, Any]], Any] | None = None,
    core_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _reload_volumes()
    Path(config["stamp_output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(expected_output).mkdir(parents=True, exist_ok=True)
    manifest_path = _stage_manifest_path(config, stage)
    fingerprint = _config_fingerprint(stage, config)

    if bool(config.get("resume", True)) and not bool(config.get("force", False)):
        completed = _completed_stage(manifest_path, fingerprint, expected_output)
        if completed is not None:
            return {
                "stage": stage,
                "status": "completed",
                "skipped": True,
                "manifest_path": manifest_path,
                "state_path": completed.get("state_path"),
                "result": completed.get("result", {}),
            }

    for path, description in required_inputs:
        _require_artifacts(path, description)

    started_at = _utc_now()
    _write_json(
        manifest_path,
        {
            "format": "imessage-stamp-stage-v1",
            "stage": stage,
            "status": "running",
            "fingerprint": fingerprint,
            "started_at": started_at,
            "run_name": config["run_name"],
            "round": config.get("round"),
            "expected_output": expected_output,
        },
    )
    artifact_volume.commit()

    try:
        if runner is not None:
            result = runner(config)
        else:
            function = _resolve_callable(module_names, callable_names)
            if function is None:  # pragma: no cover - required=True guarantees this.
                raise StampIntegrationError(f"No callable resolved for {stage}")
            invocation_config = core_config or config
            if checkpointed:
                result = _invoke_with_checkpoint_commits(
                    function,
                    invocation_config,
                    expected_output,
                )
            else:
                result = _invoke_core(
                    function,
                    invocation_config,
                    checkpointed=False,
                )
        _require_artifacts(expected_output, f"{stage} output")
        state_path = _best_state_path(result, expected_output) if checkpointed else None
        public_result = _public_summary(result)
        completed_manifest = {
            "format": "imessage-stamp-stage-v1",
            "stage": stage,
            "status": "completed",
            "fingerprint": fingerprint,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "run_name": config["run_name"],
            "round": config.get("round"),
            "expected_output": expected_output,
            "state_path": state_path,
            "result": public_result,
        }
        _write_json(manifest_path, completed_manifest)
        _commit_all()
        return {
            "stage": stage,
            "status": "completed",
            "skipped": False,
            "manifest_path": manifest_path,
            "state_path": state_path,
            "result": public_result,
        }
    except Exception as error:
        _write_json(
            manifest_path,
            {
                "format": "imessage-stamp-stage-v1",
                "stage": stage,
                "status": "failed",
                "fingerprint": fingerprint,
                "started_at": started_at,
                "failed_at": _utc_now(),
                "run_name": config["run_name"],
                "round": config.get("round"),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        _commit_all()
        raise


def _record_uploaded_inputs(config: dict[str, Any]) -> dict[str, Any]:
    _reload_volumes()
    prepared_dir = str(config["prepared_dir"])
    _require_artifacts(prepared_dir, "uploaded prepared STAMP inputs")
    train_path = Path(str(config["units_train_path"]))
    if not train_path.is_file():
        raise FileNotFoundError(
            "Prepared uploads must contain train.jsonl at "
            f"{config['units_train_path']}"
        )
    manifest_path = _stage_manifest_path(config, "prepare")
    result = {
        "prepared_dir": prepared_dir,
        "train_path": str(train_path),
        "validation_path": str(config["units_validation_path"]),
        "test_path": str(config["units_test_path"]),
        "uploaded": True,
    }
    _write_json(
        manifest_path,
        {
            "format": "imessage-stamp-stage-v1",
            "stage": "prepare",
            "status": "completed",
            "fingerprint": _config_fingerprint("prepare", config),
            "completed_at": _utc_now(),
            "run_name": config["run_name"],
            "expected_output": prepared_dir,
            "result": result,
        },
    )
    _commit_all()
    return {
        "stage": "prepare",
        "status": "completed",
        "skipped": False,
        "manifest_path": manifest_path,
        "state_path": None,
        "result": result,
    }


@app.function(**_GPU_FUNCTION_OPTIONS)
def prepare_neutralization_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Register locally prepared outgoing-only style units."""
    if bool(config.get("inputs_are_prepared", False)):
        return _record_uploaded_inputs(config)
    raise ValueError(
        "Run `imessage-download prepare-stamp` locally, then pass its output directory. "
        "Raw messages (including incoming conversation context) are never uploaded."
    )


def _neutralization_records(path: str, limit: int | None) -> list[dict[str, Any]]:
    records = _read_jsonl(path, limit=limit)
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not any(key in record for key in ("original", "target_bubbles", "reply")):
            bubbles = record.get("bubbles")
            if isinstance(bubbles, Sequence) and not isinstance(bubbles, (str, bytes)):
                record["original"] = [str(bubble) for bubble in bubbles]
            elif record.get("target"):
                record["original"] = str(record["target"]).splitlines()
        normalized.append(record)
    return normalized


def _make_qwen_neutralizer(
    config: Mapping[str, Any],
) -> Callable[[list[list[dict[str, str]]]], list[str]]:
    """Build a batched neutralizer so one forward pass covers many records."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from imessage_mlx.stamp.neutralize import token_budget_slices

    model_name = str(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only batching requires left padding so completions start together.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))

    temperature = float(config.get("neutralize_temperature", 0.2))
    retry_temperature = float(config.get("neutralize_retry_temperature", 0.9))
    max_new_tokens = int(config.get("neutralize_max_new_tokens", 512))
    top_p = float(config.get("neutralize_top_p", 0.9))
    token_budget = int(config.get("neutralize_token_budget", 16384))

    def options_for(attempt: int) -> dict[str, Any]:
        # The first pass stays near-deterministic for faithfulness. Retries must
        # sample more widely, since repeating a rejected draft wastes the attempt.
        chosen = temperature if attempt == 0 else max(temperature, retry_temperature)
        options: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if chosen > 0:
            options.update({"do_sample": True, "temperature": chosen, "top_p": top_p})
        else:
            options["do_sample"] = False
        return options

    def generate_once(prompts: list[str], generation_options: dict[str, Any]) -> list[str]:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_options)
        prompt_tokens = inputs["input_ids"].shape[1]
        return tokenizer.batch_decode(
            generated[:, prompt_tokens:],
            skip_special_tokens=True,
        )

    def generate_slice(prompts: list[str], attempt: int) -> list[str]:
        failed_size = 0
        try:
            return generate_once(prompts, options_for(attempt))
        except torch.OutOfMemoryError:
            if len(prompts) == 1:
                raise
            failed_size = len(prompts)
        # Freeing has to happen after the except block: while it is running, the
        # exception's traceback still references the failed frame and its tensors,
        # so empty_cache() would have nothing it is allowed to release.
        torch.cuda.empty_cache()
        print(f"neutralize: splitting batch of {failed_size} after OOM", flush=True)
        middle = failed_size // 2
        first = generate_slice(prompts[:middle], attempt)
        torch.cuda.empty_cache()
        return first + generate_slice(prompts[middle:], attempt)

    def generate_batch(batch: list[list[dict[str, str]]], *, attempt: int = 0) -> list[str]:
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in batch
        ]
        # Padding makes every sequence as long as the longest in its batch, so a
        # single long message would otherwise size the whole batch's KV cache.
        # Grouping by a token budget keeps peak memory flat instead of relying on
        # recovery after the allocator has already run out.
        lengths = [len(ids) + max_new_tokens for ids in tokenizer(prompts)["input_ids"]]
        outputs: list[str] = []
        for start, end in token_budget_slices(lengths, token_budget):
            outputs.extend(generate_slice(prompts[start:end], attempt))
        return outputs

    return generate_batch


def _make_neutralize_reporter(
    *,
    split: str,
    total: int,
    commit_every: int,
) -> Callable[[Mapping[str, Any]], None]:
    """Commit the artifact volume periodically and log neutralization throughput."""

    started = time.monotonic()
    state = {"last_commit": 0}

    def report(progress: Mapping[str, Any]) -> None:
        done = int(progress["generated"]) + int(progress["skipped_existing"])
        if done - state["last_commit"] < commit_every and done < total:
            return
        state["last_commit"] = done
        artifact_volume.commit()
        elapsed = max(time.monotonic() - started, 1e-6)
        rate = int(progress["generated"]) / elapsed
        remaining = f"{(total - done) / rate / 60:.1f} min" if rate > 0 else "unknown"
        print(
            f"neutralize[{split}]: {done}/{total} "
            f"({rate:.1f}/s, ~{remaining} left, failed={progress['failed']})",
            flush=True,
        )

    return report


def _make_progress_logger(
    label: str,
    total: int,
    *,
    every: int = 25,
) -> Callable[..., None]:
    """Log throughput and ETA for a loop that generates one record at a time.

    These loops run for hours and are otherwise silent, which leaves no way to
    tell slow progress apart from a hang.
    """

    started = time.monotonic()

    def report(done: int, **extra: Any) -> None:
        if done % every and done < total:
            return
        elapsed = max(time.monotonic() - started, 1e-6)
        rate = done / elapsed
        remaining = f"{(total - done) / rate / 60:.1f} min" if rate > 0 else "unknown"
        details = "".join(f" {key}={value}" for key, value in extra.items())
        print(
            f"{label}: {done}/{total} ({rate:.2f}/s, ~{remaining} left{details})",
            flush=True,
        )

    return report


def _run_neutralization_core(config: dict[str, Any]) -> Any:
    function = _resolve_callable(_NEUTRALIZE_MODULES, _NEUTRALIZE_CALLABLES)
    if function is None:  # pragma: no cover - required=True guarantees this.
        raise StampIntegrationError("No neutralization callable resolved")
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        parameters = {}
    if not {"records", "generator"}.issubset(parameters):
        return _invoke_core(function, config, checkpointed=False)

    limit_value = config.get("neutralize_limit", config.get("max_examples"))
    limit = int(limit_value) if limit_value is not None else None
    generator = _make_qwen_neutralizer(config)
    batch_size = int(config.get("neutralize_batch_size", 32))
    commit_every = max(1, int(config.get("neutralize_commit_every", 256)))
    split_paths = {
        "train": (config["units_train_path"], config["neutral_train_path"]),
        "validation": (
            config["units_validation_path"],
            config["neutral_validation_path"],
        ),
        "test": (config["units_test_path"], config["neutral_test_path"]),
    }
    supports_batching = "batch_generator" in parameters
    results: dict[str, Any] = {}
    for split, (source_path, output_path) in split_paths.items():
        source = Path(str(source_path))
        if not source.is_file():
            if split == "train":
                raise FileNotFoundError(f"Missing prepared training units: {source}")
            continue
        records = _neutralization_records(str(source), limit)
        call_options: dict[str, Any] = {
            "records": records,
            "output_path": str(output_path),
            "max_attempts": int(config.get("neutralize_max_attempts", 2)),
            "max_unchanged_ratio": float(config.get("neutralize_max_unchanged_ratio", 0.05)),
            "report_path": f"{config['neutral_dir']}/{split}-report.json",
        }
        if supports_batching:
            call_options.update(
                {
                    "batch_generator": generator,
                    "batch_size": batch_size,
                    "on_progress": _make_neutralize_reporter(
                        split=split,
                        total=len(records),
                        commit_every=commit_every,
                    ),
                }
            )
        else:
            call_options["generator"] = lambda messages: generator([messages])[0]
        results[split] = function(**call_options)
        artifact_volume.commit()
    return {"splits": results, "neutral_dir": config["neutral_dir"]}


@app.function(**_GPU_FUNCTION_OPTIONS)
def neutralize(config: dict[str, Any]) -> dict[str, Any]:
    """Generate resumable neutral/original pairs for all configured splits."""
    return _run_core_stage(
        stage="neutralize",
        config=config,
        module_names=_NEUTRALIZE_MODULES,
        callable_names=_NEUTRALIZE_CALLABLES,
        required_inputs=((str(config["units_train_path"]), "prepared training style units"),),
        expected_output=str(config["neutral_dir"]),
        runner=_run_neutralization_core,
    )


@app.function(**_GPU_FUNCTION_OPTIONS)
def train_style_classifier(config: dict[str, Any]) -> dict[str, Any]:
    """Train the binary original-vs-neutral style proxy classifier."""
    classifier_config = {
        "train_path": config["neutral_train_path"],
        "validation_path": config["neutral_validation_path"],
        "test_path": config.get("neutral_test_path"),
        "output_dir": config["classifier_dir"],
        "model_name": config.get(
            "classifier_model_name",
            "answerdotai/ModernBERT-base",
        ),
        "max_length": int(config.get("classifier_max_length", 512)),
        "epochs": float(config.get("classifier_epochs", config.get("epochs", 3.0))),
        "batch_size": int(config.get("classifier_batch_size", 16)),
        "gradient_accumulation_steps": int(
            config.get("classifier_gradient_accumulation_steps", 1)
        ),
        "learning_rate": float(config.get("classifier_learning_rate", 2e-5)),
        "weight_decay": float(config.get("classifier_weight_decay", 0.01)),
        "warmup_ratio": float(config.get("classifier_warmup_ratio", 0.1)),
        "logging_steps": int(config.get("classifier_logging_steps", 25)),
        "seed": int(config["seed"]),
        "bf16": bool(config.get("classifier_bf16", True)),
        "resume": bool(config.get("resume", True)),
    }
    return _run_core_stage(
        stage="classifier",
        config=config,
        module_names=_CLASSIFIER_MODULES,
        callable_names=_CLASSIFIER_CALLABLES,
        required_inputs=(
            (str(config["neutral_train_path"]), "neutral training pairs"),
            (str(config["neutral_validation_path"]), "neutral validation pairs"),
        ),
        expected_output=str(config["classifier_dir"]),
        checkpointed=True,
        core_config=classifier_config,
    )


@app.function(**_GPU_FUNCTION_OPTIONS)
def train_initial_sft(config: dict[str, Any]) -> dict[str, Any]:
    """Train the initial neutral-to-original LoRA adapter."""
    sft_config = {
        "train_path": config["neutral_train_path"],
        "validation_path": config["neutral_validation_path"],
        "output_dir": config["initial_sft_dir"],
        "model_name_or_path": config["model_name"],
        "max_length": int(config.get("sft_max_length", 1024)),
        "epochs": float(config.get("sft_epochs", config.get("epochs", 2.0))),
        "batch_size": int(config.get("sft_batch_size", 2)),
        "gradient_accumulation_steps": int(
            config.get("sft_gradient_accumulation_steps", 8)
        ),
        "learning_rate": float(config.get("sft_learning_rate", 2e-4)),
        "warmup_ratio": float(config.get("sft_warmup_ratio", 0.05)),
        "weight_decay": float(config.get("sft_weight_decay", 0.01)),
        "logging_steps": int(config.get("sft_logging_steps", 10)),
        "save_steps": int(config.get("sft_save_steps", config.get("save_steps", 100))),
        "save_total_limit": int(config.get("sft_save_total_limit", 2)),
        "seed": int(config["seed"]),
        "bf16": True,
        "gradient_checkpointing": True,
        "lora_rank": int(config.get("lora_rank", 16)),
        "lora_alpha": int(config.get("lora_alpha", 32)),
        "lora_dropout": float(config.get("lora_dropout", 0.05)),
        "lora_target_modules": config.get("lora_target_modules", "all-linear"),
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "resume": bool(config.get("resume", True)),
    }
    return _run_core_stage(
        stage="initial-sft",
        config=config,
        module_names=_SFT_MODULES,
        callable_names=_SFT_CALLABLES,
        required_inputs=((str(config["neutral_train_path"]), "neutral training pairs"),),
        expected_output=str(config["initial_sft_dir"]),
        checkpointed=True,
        core_config=sft_config,
    )


def _load_rewrite_model(
    model_name: str,
    state_path: str,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    state = Path(state_path)
    model_source = state_path
    tokenizer_source = state_path
    if state.is_dir() and (state / "adapter_config.json").is_file():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(state_path)
        model_source = str(
            getattr(peft_config, "base_model_name_or_path", "") or model_name
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base_model, state_path).merge_and_unload()
    else:
        if not state.exists():
            model_source = state_path or model_name
            tokenizer_source = model_source
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
    if not (state / "tokenizer_config.json").is_file():
        tokenizer_source = model_source
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def _generate_rewrite_candidates(
    *,
    model: Any,
    tokenizer: Any,
    neutral_bubbles: Sequence[str],
    count: int,
    seed: int,
    config: Mapping[str, Any],
    fallback_bubbles: Sequence[Sequence[str]] = (),
) -> list[list[str]]:
    import torch

    from imessage_mlx.stamp.sft import (
        build_rewrite_messages,
        parse_rewrite_output,
        validate_rewrite,
    )

    messages = build_rewrite_messages(neutral_bubbles)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    candidates: list[list[str]] = []
    seen: set[str] = set()
    for attempt in range(3):
        torch.manual_seed(seed + attempt)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + attempt)
        temperature = float(config.get("candidate_temperature", 0.8))
        generation_options: dict[str, Any] = {
            "max_new_tokens": int(config.get("candidate_max_new_tokens", 256)),
            "num_return_sequences": count,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_options.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": float(config.get("candidate_top_p", 0.95)),
                }
            )
        else:
            generation_options["do_sample"] = False
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_options)
        prompt_tokens = inputs["input_ids"].shape[1]
        for output in outputs:
            decoded = tokenizer.decode(output[prompt_tokens:], skip_special_tokens=True)
            try:
                bubbles = parse_rewrite_output(
                    decoded,
                )
            except (TypeError, ValueError):
                continue
            validation = validate_rewrite(neutral_bubbles, bubbles)
            key = json.dumps(bubbles, ensure_ascii=False, separators=(",", ":"))
            if validation.valid and key not in seen:
                seen.add(key)
                candidates.append(bubbles)
                if len(candidates) >= count:
                    return candidates
    for fallback in fallback_bubbles:
        normalized = [str(bubble) for bubble in fallback]
        validation = validate_rewrite(neutral_bubbles, normalized)
        key = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
        if validation.valid and key not in seen:
            seen.add(key)
            candidates.append(normalized)
            if len(candidates) >= count:
                return candidates
    if len(candidates) < min(2, count):
        raise ValueError(
            "Candidate generation produced fewer than two distinct content-valid rewrites"
        )
    return candidates


def _load_reward_scorers(config: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    classifier_dir = Path(str(config["classifier_dir"]))
    classifier_source = (
        classifier_dir / "final"
        if (classifier_dir / "final").is_dir()
        else classifier_dir
    )
    classifier_tokenizer = AutoTokenizer.from_pretrained(
        str(classifier_source),
        use_fast=True,
    )
    classifier = AutoModelForSequenceClassification.from_pretrained(
        str(classifier_source),
        dtype=torch.bfloat16,
        device_map="auto",
    )
    classifier.eval()
    semantic_model = SentenceTransformer(
        str(config.get("semantic_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        device="cuda",
    )
    base_tokenizer = AutoTokenizer.from_pretrained(str(config["model_name"]), use_fast=True)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        str(config["model_name"]),
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    base_model.eval()
    return (
        classifier,
        classifier_tokenizer,
        semantic_model,
        base_model,
        base_tokenizer,
    )


def _score_rewrites(
    *,
    pair_id: str,
    source_text: str,
    candidate_texts: Sequence[str],
    scorers: tuple[Any, Any, Any, Any, Any],
    config: Mapping[str, Any],
) -> list[Any]:
    import torch

    from imessage_mlx.stamp.rewards import candidate_reward_from_scores

    classifier, classifier_tokenizer, semantic_model, base_model, base_tokenizer = scorers
    classifier_inputs = classifier_tokenizer(
        list(candidate_texts),
        padding=True,
        truncation=True,
        max_length=int(config.get("classifier_max_length", 512)),
        return_tensors="pt",
    ).to(classifier.device)
    with torch.inference_mode():
        style_probabilities = torch.softmax(
            classifier(**classifier_inputs).logits.float(),
            dim=-1,
        )[:, 1].tolist()

    source_embeddings = semantic_model.encode(
        [source_text] * len(candidate_texts),
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    candidate_embeddings = semantic_model.encode(
        list(candidate_texts),
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    semantic_similarities = (
        (source_embeddings * candidate_embeddings).sum(dim=1).clamp(0, 1).tolist()
    )

    rewards = []
    for index, (text, style_probability, semantic_similarity) in enumerate(
        zip(
            candidate_texts,
            style_probabilities,
            semantic_similarities,
            strict=True,
        )
    ):
        likelihood_inputs = base_tokenizer(
            text,
            truncation=True,
            max_length=int(config.get("likelihood_max_length", 512)),
            return_tensors="pt",
        ).to(base_model.device)
        token_count = max(1, int(likelihood_inputs["input_ids"].shape[1]) - 1)
        with torch.inference_mode():
            loss = base_model(
                **likelihood_inputs,
                labels=likelihood_inputs["input_ids"],
            ).loss
        candidate_id = hashlib.sha256(
            f"{pair_id}\0{index}\0{text}".encode()
        ).hexdigest()[:24]
        rewards.append(
            candidate_reward_from_scores(
                candidate_id=candidate_id,
                text=text,
                style_probability=float(style_probability),
                semantic_similarity=float(semantic_similarity),
                base_log_likelihood=-float(loss.item()) * token_count,
                token_count=token_count,
                source_length=max(1, len(source_text)),
            )
        )
    return rewards


def _generate_and_score_locally(config: dict[str, Any]) -> dict[str, Any]:
    import gc

    import torch

    from imessage_mlx.stamp.preference import make_cpo_pair
    from imessage_mlx.stamp.rewards import (
        dynamic_reward_exponents,
        select_hope_and_fear,
    )

    limit_value = config.get("preference_limit", config.get("max_examples"))
    limit = int(limit_value) if limit_value is not None else None
    records = _read_jsonl(config["neutral_train_path"], limit=limit)
    if not records:
        raise ValueError("Preference generation requires neutral training pairs")

    generation_model, generation_tokenizer = _load_rewrite_model(
        str(config["model_name"]),
        str(config["previous_state_path"]),
    )
    scorers = _load_reward_scorers(config)
    candidate_groups: list[dict[str, Any]] = []
    skipped_pairs: list[dict[str, str]] = []
    round_label = _round_name(int(config["round"]))
    print(
        f"preferences[{round_label}]: sampling {config['candidates_per_input']} "
        f"candidates for {len(records)} pairs",
        flush=True,
    )
    progress = _make_progress_logger(f"preferences[{round_label}]", len(records))
    for index, record in enumerate(records):
        pair_id = str(record.get("pair_id", ""))
        neutral_value = record.get("neutral")
        if (
            not pair_id
            or not isinstance(neutral_value, Sequence)
            or isinstance(neutral_value, (str, bytes))
        ):
            raise ValueError("Neutral pairs require pair_id and neutral bubbles")
        neutral_bubbles = [str(bubble) for bubble in neutral_value]
        original_value = record.get("original")
        if not isinstance(original_value, Sequence) or isinstance(original_value, (str, bytes)):
            raise ValueError("Neutral pairs require original bubbles")
        original_bubbles = [str(bubble) for bubble in original_value]
        record_seed = int(config["seed"]) + int(
            hashlib.sha256(pair_id.encode()).hexdigest()[:8],
            16,
        )
        try:
            bubbles = _generate_rewrite_candidates(
                model=generation_model,
                tokenizer=generation_tokenizer,
                neutral_bubbles=neutral_bubbles,
                count=int(config["candidates_per_input"]),
                seed=record_seed + index,
                config=config,
                fallback_bubbles=(neutral_bubbles, original_bubbles),
            )
        except ValueError as error:
            # A pair the model cannot rewrite twice yields no preference, but it
            # says nothing about the rest of the corpus, so drop it and continue.
            skipped_pairs.append({"pair_id": pair_id, "reason": str(error)})
        else:
            texts = ["\n".join(candidate) for candidate in bubbles]
            rewards = _score_rewrites(
                pair_id=pair_id,
                source_text="\n".join(neutral_bubbles),
                candidate_texts=texts,
                scorers=scorers,
                config=config,
            )
            candidate_groups.append(
                {
                    "pair_id": pair_id,
                    "neutral": neutral_bubbles,
                    "bubbles": bubbles,
                    "rewards": rewards,
                }
            )
        progress(index + 1, kept=len(candidate_groups), skipped=len(skipped_pairs))

    if skipped_pairs:
        print(
            f"Skipped {len(skipped_pairs)}/{len(records)} pairs lacking two distinct "
            f"candidates; first was {skipped_pairs[0]['pair_id']}",
            flush=True,
        )
    minimum_yield = float(config.get("min_preference_yield", 0.5))
    if len(candidate_groups) < max(1, int(minimum_yield * len(records))):
        raise ValueError(
            f"Only {len(candidate_groups)} of {len(records)} pairs produced candidate "
            f"groups, under the {minimum_yield:.0%} floor; the rewrite model looks degenerate"
        )

    initial_selections = [
        select_hope_and_fear(group["rewards"]) for group in candidate_groups
    ]
    exponents = dynamic_reward_exponents(
        [(selection.hope, selection.fear) for selection in initial_selections]
    )
    preference_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for group in candidate_groups:
        selection = select_hope_and_fear(group["rewards"], exponents=exponents)
        by_id = {
            reward.candidate_id: bubbles
            for reward, bubbles in zip(
                group["rewards"],
                group["bubbles"],
                strict=True,
            )
        }
        preference_records.append(
            make_cpo_pair(
                pair_id=group["pair_id"],
                neutral_bubbles=group["neutral"],
                chosen_bubbles=by_id[selection.hope.candidate_id],
                rejected_bubbles=by_id[selection.fear.candidate_id],
                metadata={
                    "round": config["round"],
                    "hope_candidate_id": selection.hope.candidate_id,
                    "fear_candidate_id": selection.fear.candidate_id,
                    "reward_exponents": exponents.as_dict(),
                },
            )
        )
        candidate_records.append(
            {
                "pair_id": group["pair_id"],
                "round": config["round"],
                "neutral": group["neutral"],
                "candidates": [
                    {
                        **reward.as_dict(exponents),
                        "bubbles": bubbles,
                    }
                    for reward, bubbles in zip(
                        group["rewards"],
                        group["bubbles"],
                        strict=True,
                    )
                ],
            }
        )

    validation_records = [
        record
        for record in preference_records
        if int(hashlib.sha256(str(record["pair_id"]).encode()).hexdigest()[:8], 16) % 20 == 0
    ]
    if preference_records and not validation_records:
        validation_records = [preference_records[-1]]
    validation_ids = {str(record["pair_id"]) for record in validation_records}
    training_records = [
        record for record in preference_records if str(record["pair_id"]) not in validation_ids
    ]
    if not training_records:
        raise ValueError("Preference generation needs at least two pairs for train/validation")

    _write_jsonl(config["candidates_path"], candidate_records)
    _write_jsonl(config["preferences_path"], training_records)
    _write_jsonl(config["preferences_validation_path"], validation_records)
    _write_json(
        f"{config['round_dir']}/preference-summary.json",
        {
            "format": "imessage-stamp-preference-round-v1",
            "round": config["round"],
            "pairs": len(preference_records),
            "train_pairs": len(training_records),
            "validation_pairs": len(validation_records),
            "skipped_pairs": len(skipped_pairs),
            "candidates_per_input": config["candidates_per_input"],
            "reward_exponents": exponents.as_dict(),
            "previous_state_path": config["previous_state_path"],
        },
    )
    del generation_model
    del scorers
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "candidates_path": config["candidates_path"],
        "preferences_path": config["preferences_path"],
        "preferences_validation_path": config["preferences_validation_path"],
        "pairs": len(preference_records),
        "reward_exponents": exponents.as_dict(),
    }


def _run_preference_core(config: dict[str, Any]) -> Any:
    combined = _resolve_callable(
        _PREFERENCE_MODULES,
        _PREFERENCE_CALLABLES,
        required=False,
    )
    if combined is not None:
        return _invoke_core(combined, config, checkpointed=False)

    generate = _resolve_callable(
        _PREFERENCE_MODULES,
        _GENERATION_CALLABLES,
        required=False,
    )
    score = _resolve_callable(
        _PREFERENCE_MODULES,
        _SCORING_CALLABLES,
        required=False,
    )
    if generate is None or score is None:
        return _generate_and_score_locally(config)
    generated = _invoke_core(generate, config, checkpointed=False)

    scoring_config = dict(config)
    generated_path = _find_result_path(
        generated,
        ("candidates_path", "candidate_path", "output_path"),
    )
    if generated_path:
        scoring_config["candidates_path"] = generated_path
        scoring_config["candidate_path"] = generated_path
    scored = _invoke_core(score, scoring_config, checkpointed=False)
    return {"generation": generated, "scoring": scored}


@app.function(**_GPU_FUNCTION_OPTIONS)
def generate_and_score_preferences(config: dict[str, Any]) -> dict[str, Any]:
    """Generate candidates from the current state and select hope/fear pairs."""
    round_name = _round_name(int(config["round"]))
    stage = f"{round_name}-preferences"
    output_dir = str(config["round_dir"])
    _reload_volumes()
    Path(config["stamp_output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = _stage_manifest_path(config, stage)
    fingerprint = _config_fingerprint(stage, config)

    if bool(config.get("resume", True)) and not bool(config.get("force", False)):
        completed = _completed_stage(manifest_path, fingerprint, output_dir)
        if completed is not None:
            return {
                "stage": stage,
                "status": "completed",
                "skipped": True,
                "manifest_path": manifest_path,
                "state_path": config["previous_state_path"],
                "result": completed.get("result", {}),
            }

    _require_file(config["neutral_train_path"], "neutral training pairs")
    _require_artifacts(config["previous_state_path"], "previous model/adapter state")
    started_at = _utc_now()
    _write_json(
        manifest_path,
        {
            "format": "imessage-stamp-stage-v1",
            "stage": stage,
            "status": "running",
            "fingerprint": fingerprint,
            "started_at": started_at,
            "run_name": config["run_name"],
            "round": config["round"],
            "previous_state_path": config["previous_state_path"],
            "expected_output": output_dir,
        },
    )
    artifact_volume.commit()
    try:
        result = _run_preference_core(config)
        _require_artifacts(output_dir, "preference generation and scoring output")
        public_result = _public_summary(result)
        _write_json(
            manifest_path,
            {
                "format": "imessage-stamp-stage-v1",
                "stage": stage,
                "status": "completed",
                "fingerprint": fingerprint,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "run_name": config["run_name"],
                "round": config["round"],
                "previous_state_path": config["previous_state_path"],
                "expected_output": output_dir,
                "result": public_result,
            },
        )
        _commit_all()
        return {
            "stage": stage,
            "status": "completed",
            "skipped": False,
            "manifest_path": manifest_path,
            "state_path": config["previous_state_path"],
            "result": public_result,
        }
    except Exception as error:
        _write_json(
            manifest_path,
            {
                "format": "imessage-stamp-stage-v1",
                "stage": stage,
                "status": "failed",
                "fingerprint": fingerprint,
                "started_at": started_at,
                "failed_at": _utc_now(),
                "run_name": config["run_name"],
                "round": config["round"],
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        _commit_all()
        raise


@app.function(**_GPU_FUNCTION_OPTIONS)
def train_cpo_round(config: dict[str, Any]) -> dict[str, Any]:
    """Train one CPO LoRA from the preceding SFT/CPO model state."""
    round_name = _round_name(int(config["round"]))

    def train_from_previous_state(stage_config: dict[str, Any]) -> Any:
        import gc

        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        previous_state = Path(str(stage_config["previous_state_path"]))
        merged_dir = Path(str(stage_config["round_dir"])) / "input-model"
        if not (merged_dir / "config.json").is_file():
            adapter_config = previous_state / "adapter_config.json"
            if not adapter_config.is_file():
                _require_file(
                    previous_state / "config.json",
                    "previous merged model configuration",
                )
                merged_model_path = str(previous_state)
            else:
                peft_config = PeftConfig.from_pretrained(str(previous_state))
                base_path = str(
                    getattr(peft_config, "base_model_name_or_path", "")
                    or stage_config["model_name"]
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_path,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    attn_implementation="sdpa",
                )
                merged_model = PeftModel.from_pretrained(base_model, str(previous_state))
                merged_model = merged_model.merge_and_unload()
                merged_dir.mkdir(parents=True, exist_ok=True)
                merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
                tokenizer_source = (
                    str(previous_state)
                    if (previous_state / "tokenizer_config.json").is_file()
                    else base_path
                )
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
                tokenizer.save_pretrained(str(merged_dir))
                _commit_checkpoint()
                merged_model_path = str(merged_dir)
                del merged_model, base_model
                gc.collect()
                torch.cuda.empty_cache()
        else:
            merged_model_path = str(merged_dir)

        function = _resolve_callable(_CPO_MODULES, _CPO_CALLABLES)
        if function is None:  # pragma: no cover - required=True guarantees this.
            raise StampIntegrationError("No CPO training callable resolved")
        cpo_config = {
            "train_path": stage_config["preferences_path"],
            "validation_path": stage_config.get("preferences_validation_path"),
            "output_dir": stage_config["cpo_dir"],
            "model_name_or_path": merged_model_path,
            "max_length": int(stage_config.get("cpo_max_length", 1024)),
            "epochs": float(
                stage_config.get("cpo_epochs", stage_config.get("epochs", 1.0))
            ),
            "batch_size": int(stage_config.get("cpo_batch_size", 1)),
            "gradient_accumulation_steps": int(
                stage_config.get("cpo_gradient_accumulation_steps", 8)
            ),
            "learning_rate": float(stage_config.get("cpo_learning_rate", 5e-6)),
            "warmup_ratio": float(stage_config.get("cpo_warmup_ratio", 0.05)),
            "weight_decay": float(stage_config.get("cpo_weight_decay", 0.01)),
            "logging_steps": int(stage_config.get("cpo_logging_steps", 10)),
            "save_steps": int(
                stage_config.get("cpo_save_steps", stage_config.get("save_steps", 100))
            ),
            "save_total_limit": int(stage_config.get("cpo_save_total_limit", 2)),
            "seed": int(stage_config["seed"]),
            "bf16": True,
            "gradient_checkpointing": True,
            "beta": float(stage_config.get("cpo_beta", 0.1)),
            "cpo_alpha": float(stage_config.get("cpo_alpha", 1.0)),
            "loss_type": str(stage_config.get("cpo_loss_type", "sigmoid")),
            "lora_rank": int(stage_config.get("lora_rank", 16)),
            "lora_alpha": int(stage_config.get("lora_alpha", 32)),
            "lora_dropout": float(stage_config.get("lora_dropout", 0.05)),
            "lora_target_modules": stage_config.get(
                "lora_target_modules",
                "all-linear",
            ),
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "resume": bool(stage_config.get("resume", True)),
        }
        return _invoke_with_checkpoint_commits(
            function,
            cpo_config,
            str(stage_config["cpo_dir"]),
        )

    return _run_core_stage(
        stage=f"{round_name}-cpo",
        config=config,
        module_names=_CPO_MODULES,
        callable_names=_CPO_CALLABLES,
        required_inputs=(
            (str(config["preferences_path"]), "hope/fear preference pairs"),
            (str(config["previous_state_path"]), "previous model/adapter state"),
        ),
        expected_output=str(config["cpo_dir"]),
        checkpointed=True,
        runner=train_from_previous_state,
    )


@app.function(**_GPU_FUNCTION_OPTIONS)
def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the base model, initial SFT, and completed CPO rounds."""
    required_inputs: list[tuple[str, str]] = [
        (str(config["neutral_test_path"]), "untouched neutral test pairs"),
        (str(config["classifier_dir"]), "style classifier"),
        (str(config["initial_state_path"]), "initial SFT model/adapter state"),
    ]
    required_inputs.extend(
        (str(path), f"CPO round {index} model/adapter state")
        for index, path in enumerate(config["round_state_paths"], start=1)
    )

    def generate_and_evaluate(stage_config: dict[str, Any]) -> Any:
        import gc

        import torch

        limit_value = stage_config.get(
            "evaluation_limit",
            stage_config.get("max_examples"),
        )
        limit = int(limit_value) if limit_value is not None else None
        records = _read_jsonl(stage_config["neutral_test_path"], limit=limit)
        if not records:
            raise ValueError("Evaluation requires neutral test pairs")

        scorers = _load_reward_scorers(stage_config)
        named_rows: dict[str, list[dict[str, Any]]] = {}
        for model_index, (name, state_path) in enumerate(
            stage_config["model_states"].items()
        ):
            model, tokenizer = _load_rewrite_model(
                str(stage_config["model_name"]),
                str(state_path),
            )
            rows: list[dict[str, Any]] = []
            print(f"evaluate[{name}]: rewriting {len(records)} test pairs", flush=True)
            progress = _make_progress_logger(f"evaluate[{name}]", len(records))
            for record_index, record in enumerate(records):
                pair_id = str(record.get("pair_id", ""))
                neutral_value = record.get("neutral")
                original_value = record.get("original")
                if (
                    not pair_id
                    or not isinstance(neutral_value, Sequence)
                    or isinstance(neutral_value, (str, bytes))
                    or not isinstance(original_value, Sequence)
                    or isinstance(original_value, (str, bytes))
                ):
                    raise ValueError(
                        "Evaluation pairs require pair_id, neutral, and original bubbles"
                    )
                neutral_bubbles = [str(bubble) for bubble in neutral_value]
                original_bubbles = [str(bubble) for bubble in original_value]
                generated_bubbles = _generate_rewrite_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    neutral_bubbles=neutral_bubbles,
                    count=1,
                    seed=(
                        int(stage_config["seed"])
                        + model_index * 100_000
                        + record_index
                    ),
                    config={**stage_config, "candidate_temperature": 0.0},
                    fallback_bubbles=(neutral_bubbles,),
                )[0]
                generated_text = "\n".join(generated_bubbles)
                reward = _score_rewrites(
                    pair_id=pair_id,
                    source_text="\n".join(neutral_bubbles),
                    candidate_texts=[generated_text],
                    scorers=scorers,
                    config=stage_config,
                )[0]
                rows.append(
                    {
                        "pair_id": pair_id,
                        "neutral": neutral_bubbles,
                        "original": original_bubbles,
                        "generated": generated_bubbles,
                        **reward.as_dict(),
                    }
                )
                progress(record_index + 1)
            named_rows[str(name)] = rows
            safe_name = _safe_run_name(str(name))
            _write_jsonl(
                f"{stage_config['evaluation_dir']}/{safe_name}-candidates.jsonl",
                rows,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()

        function = _resolve_callable(_EVALUATE_MODULES, _EVALUATE_CALLABLES)
        if function is None:  # pragma: no cover - required=True guarantees this.
            raise StampIntegrationError("No evaluation callable resolved")
        parameters = inspect.signature(function).parameters
        output_path = f"{stage_config['evaluation_dir']}/metrics.json"
        if "named_rows" in parameters:
            result = function(named_rows=named_rows, output_path=output_path)
        else:
            evaluation_config = dict(stage_config)
            evaluation_config["named_rows"] = named_rows
            evaluation_config["output_path"] = output_path
            result = _invoke_core(function, evaluation_config, checkpointed=False)
        model_reports = result.get("models", {}) if isinstance(result, Mapping) else {}
        report_metrics: dict[str, dict[str, Any]] = {}
        for name, model_report in model_reports.items():
            objectives = (
                model_report.get("mean_objectives", {})
                if isinstance(model_report, Mapping)
                else {}
            )
            report_metrics[str(name)] = {
                "style_probability": objectives.get("style"),
                "semantic_similarity": objectives.get("semantic"),
                "fluency": objectives.get("likelihood"),
                "reward": (
                    model_report.get("mean_aggregate_reward")
                    if isinstance(model_report, Mapping)
                    else None
                ),
            }

        method_names = list(named_rows)
        rows_by_method = {
            name: {str(row["pair_id"]): row for row in rows}
            for name, rows in named_rows.items()
        }
        first_rows = named_rows[method_names[0]]
        examples = []
        for row in first_rows:
            pair_id = str(row["pair_id"])
            examples.append(
                {
                    "pair_id": pair_id,
                    "neutral": "\n".join(str(value) for value in row["neutral"]),
                    "target": "\n".join(str(value) for value in row["original"]),
                    "outputs": {
                        name: "\n".join(
                            str(value) for value in rows_by_method[name][pair_id]["generated"]
                        )
                        for name in method_names
                    },
                    "metrics": {
                        name: {
                            "style_probability": rows_by_method[name][pair_id].get(
                                "style_probability"
                            ),
                            "semantic_similarity": rows_by_method[name][pair_id].get(
                                "semantic_similarity"
                            ),
                            "fluency": rows_by_method[name][pair_id].get(
                                "base_model_likelihood"
                            ),
                            "reward": rows_by_method[name][pair_id].get(
                                "aggregate_reward"
                            ),
                        }
                        for name in method_names
                    },
                }
            )
        classifier_summary_path = Path(stage_config["classifier_dir"]) / "training-summary.json"
        classifier_metrics = None
        if classifier_summary_path.is_file():
            classifier_summary = json.loads(
                classifier_summary_path.read_text(encoding="utf-8")
            )
            classifier_metrics = classifier_summary.get("test_metrics") or classifier_summary.get(
                "heldout_metrics"
            )
        evaluation_path = f"{stage_config['evaluation_dir']}/evaluation.json"
        _write_json(
            evaluation_path,
            {
                "format": "imessage-stamp-comparison-v1",
                "run_name": stage_config["run_name"],
                "model_name": stage_config["model_name"],
                "methods": method_names,
                "metrics": report_metrics,
                "classifier_metrics": classifier_metrics,
                "examples": examples,
                "metric_note": (
                    "Style probability is an in-domain original-vs-neutral proxy, "
                    "not proof of authorship."
                ),
            },
        )
        del scorers
        gc.collect()
        torch.cuda.empty_cache()
        return {"metrics": result, "evaluation_path": evaluation_path}

    return _run_core_stage(
        stage="evaluate",
        config=config,
        module_names=_EVALUATE_MODULES,
        callable_names=_EVALUATE_CALLABLES,
        required_inputs=required_inputs,
        expected_output=str(config["evaluation_dir"]),
        runner=generate_and_evaluate,
    )


def _base_config(
    *,
    run_name: str,
    model_name: str,
    rounds: int,
    candidates: int,
    seed: int,
    limit: int,
    smoke: bool,
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if candidates < 2:
        raise ValueError("candidates must be at least 2")
    if limit < 0:
        raise ValueError("limit cannot be negative")

    if smoke:
        rounds = 1
        candidates = 2
        limit = min(limit, 8) if limit else 8

    data_root, output_root = _stamp_roots(run_name)
    prepared_dir = f"{data_root}/prepared"
    neutral_dir = f"{output_root}/neutralization"
    classifier_dir = f"{output_root}/classifier"
    initial_sft_dir = f"{output_root}/initial-sft"
    evaluation_dir = f"{output_root}/evaluation"
    config: dict[str, Any] = {
        "format": "imessage-stamp-run-v1",
        "run_name": run_name,
        "model_name": model_name,
        "base_model_name": model_name,
        "rounds": rounds,
        "num_rounds": rounds,
        "candidates": candidates,
        "num_candidates": candidates,
        "candidates_per_input": candidates,
        "seed": seed,
        "limit": limit or None,
        "max_examples": limit or None,
        "smoke": smoke,
        "resume": resume,
        "force": force,
        "stamp_data_dir": data_root,
        "data_dir": data_root,
        "stamp_output_dir": output_root,
        "output_root": output_root,
        "cache_dir": "/cache",
        "model_cache_dir": "/cache/huggingface",
        "prepared_dir": prepared_dir,
        "sft_train_path": f"{data_root}/input/sft/train.jsonl",
        "sft_validation_path": f"{data_root}/input/sft/validation.jsonl",
        "units_train_path": f"{prepared_dir}/train.jsonl",
        "units_validation_path": f"{prepared_dir}/validation.jsonl",
        "units_test_path": f"{prepared_dir}/test.jsonl",
        "train_units_path": f"{prepared_dir}/train.jsonl",
        "validation_units_path": f"{prepared_dir}/validation.jsonl",
        "test_units_path": f"{prepared_dir}/test.jsonl",
        "neutral_dir": neutral_dir,
        "neutral_train_path": f"{neutral_dir}/train.jsonl",
        "neutral_validation_path": f"{neutral_dir}/validation.jsonl",
        "neutral_test_path": f"{neutral_dir}/test.jsonl",
        "train_path": f"{neutral_dir}/train.jsonl",
        "validation_path": f"{neutral_dir}/validation.jsonl",
        "test_path": f"{neutral_dir}/test.jsonl",
        "classifier_dir": classifier_dir,
        "classifier_output_dir": classifier_dir,
        "initial_sft_dir": initial_sft_dir,
        "sft_output_dir": initial_sft_dir,
        "evaluation_dir": evaluation_dir,
        "evaluation_output_dir": evaluation_dir,
        "merge_input_adapter": True,
        "merge_adapter": True,
    }
    if smoke:
        config.update(
            {
                "prepare_limit": limit,
                "neutralize_limit": min(limit, 4),
                "classifier_limit": limit,
                "sft_limit": min(limit, 4),
                "preference_limit": min(limit, 2),
                "evaluation_limit": min(limit, 2),
                "max_steps": 2,
                "save_steps": 1,
                "epochs": 0.01,
                "num_train_epochs": 0.01,
            }
        )
    return config


def _round_config(
    base_config: Mapping[str, Any],
    round_number: int,
    previous_state_path: str,
    *,
    preference_limit: int | None = None,
) -> dict[str, Any]:
    round_name = _round_name(round_number)
    round_dir = f"{base_config['stamp_output_dir']}/rounds/{round_name}"
    cpo_dir = f"{round_dir}/cpo"
    preferences_path = f"{round_dir}/preferences.jsonl"
    preferences_validation_path = f"{round_dir}/preferences-validation.jsonl"
    config = dict(base_config)
    config.update(
        {
            "round": round_number,
            "round_number": round_number,
            "round_index": round_number - 1,
            "round_dir": round_dir,
            "candidate_output_path": f"{round_dir}/candidates.jsonl",
            "candidates_path": f"{round_dir}/candidates.jsonl",
            "preference_output_path": preferences_path,
            "preferences_path": preferences_path,
            "preference_path": preferences_path,
            "preferences_validation_path": preferences_validation_path,
            "previous_state_path": previous_state_path,
            "previous_model_path": previous_state_path,
            "previous_adapter_dir": previous_state_path,
            "input_model_path": previous_state_path,
            "adapter_dir": previous_state_path,
            "cpo_dir": cpo_dir,
            "cpo_output_dir": cpo_dir,
            "output_dir": cpo_dir,
        }
    )
    # Round-local so the earlier stages keep the fingerprints they completed
    # under; putting this in the base config would re-run neutralize and SFT.
    if preference_limit is not None:
        config["preference_limit"] = preference_limit
    return config


def _default_state_path(config: Mapping[str, Any], round_number: int) -> str:
    if round_number == 1:
        return f"{config['initial_sft_dir']}/final"
    previous_name = _round_name(round_number - 1)
    return f"{config['stamp_output_dir']}/rounds/{previous_name}/cpo/final"


def _upload_private_inputs(
    source_path: str,
    run_name: str,
) -> tuple[str, bool]:
    source = Path(source_path).expanduser().resolve()
    data_root, _ = _stamp_roots(run_name)
    if source.is_file():
        raise ValueError(
            "source_path must be the outgoing-only directory created by "
            "`imessage-download prepare-stamp`; raw message files are not uploaded"
        )

    if source.is_dir():
        files = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        )
        if not files:
            raise FileNotFoundError(f"No JSON/JSONL prepared inputs found in {source}")
        with data_volume.batch_upload(force=True) as batch:
            for path in files:
                relative = path.relative_to(source)
                batch.put_file(path, f"/{run_name}/stamp/prepared/{relative.as_posix()}")
        return f"{data_root}/prepared", True

    raise FileNotFoundError(f"Missing source messages or prepared input directory: {source}")


def _upload_sft_splits(sft_dir: str, run_name: str) -> tuple[str, str]:
    source = Path(sft_dir).expanduser().resolve()
    train_path = source / "train.jsonl"
    validation_path = source / "validation.jsonl"
    _require_file(train_path, "local SFT training split")
    _require_file(validation_path, "local SFT validation split")
    remote_root = f"/{run_name}/stamp/input/sft"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(train_path, f"{remote_root}/train.jsonl")
        batch.put_file(validation_path, f"{remote_root}/validation.jsonl")
    return (
        f"/data{remote_root}/train.jsonl",
        f"/data{remote_root}/validation.jsonl",
    )


def _call_stage(function: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Invoke a stage Function from the remote orchestrator and validate its result."""

    print(f"Calling {function.tag}", flush=True)
    result = function.remote(config)
    if not isinstance(result, dict):
        raise TypeError(f"Expected stage result object, received {type(result).__name__}")
    print(
        f"Finished {function.tag}: status={result.get('status')} skipped={result.get('skipped')}",
        flush=True,
    )
    return result


def _execute_stamp_command(
    *,
    command: str,
    config: dict[str, Any],
    prepare_config: dict[str, Any] | None = None,
    round_number: int = 1,
    preference_limit: int | None = None,
    call_stage: Callable[[Any, dict[str, Any]], dict[str, Any]] = _call_stage,
) -> dict[str, Any]:
    """Run one STAMP command after inputs are already on Modal volumes.

    Living inside Modal (via ``orchestrate_stamp``) means later stages keep
    chaining even if the local ``modal run --detach`` client disconnects.
    """

    results: dict[str, Any] = {"command": command}
    model_name = str(config["model_name"])

    if command in {"prepare", "run"}:
        if prepare_config is None:
            raise ValueError("prepare_config is required for prepare/run")
        results["prepare"] = call_stage(prepare_neutralization_inputs, prepare_config)
        if command == "prepare":
            return results

    if command in {"neutralize", "run"}:
        results["neutralize"] = call_stage(neutralize, config)
        if command == "neutralize":
            return results

    if command in {"classifier", "run"}:
        results["classifier"] = call_stage(train_style_classifier, config)
        if command == "classifier":
            return results

    if command in {"sft", "run"}:
        sft_result = call_stage(train_initial_sft, config)
        results["sft"] = sft_result
        if command == "sft":
            return results
        current_state = str(
            sft_result.get("state_path") or f"{config['initial_sft_dir']}/final"
        )
    else:
        current_state = _default_state_path(config, round_number)

    if command in {"preferences", "cpo"}:
        if round_number > int(config["rounds"]):
            raise ValueError(
                f"round_number {round_number} exceeds configured rounds {config['rounds']}"
            )
        previous_state = _default_state_path(config, round_number)
        round_config = _round_config(
            config,
            round_number,
            previous_state,
            preference_limit=preference_limit,
        )
        if command == "preferences":
            results["preferences"] = call_stage(
                generate_and_score_preferences,
                round_config,
            )
            return results
        results["cpo"] = call_stage(train_cpo_round, round_config)
        return results

    round_states: list[str] = []
    if command == "run":
        for current_round in range(1, int(config["rounds"]) + 1):
            round_config = _round_config(
                config,
                current_round,
                current_state,
                preference_limit=preference_limit,
            )
            results[f"preferences-{_round_name(current_round)}"] = call_stage(
                generate_and_score_preferences,
                round_config,
            )
            cpo_result = call_stage(train_cpo_round, round_config)
            results[f"cpo-{_round_name(current_round)}"] = cpo_result
            current_state = str(
                cpo_result.get("state_path")
                or f"{round_config['cpo_dir']}/final"
            )
            round_states.append(current_state)
    else:
        round_states = [
            f"{config['stamp_output_dir']}/rounds/{_round_name(index)}/cpo/final"
            for index in range(1, int(config["rounds"]) + 1)
        ]

    if command in {"evaluate", "run"}:
        evaluation_config = dict(config)
        evaluation_config.update(
            {
                "initial_state_path": f"{config['initial_sft_dir']}/final",
                "initial_adapter_dir": f"{config['initial_sft_dir']}/final",
                "round_state_paths": round_states,
                "adapter_dirs": [
                    f"{config['initial_sft_dir']}/final",
                    *round_states,
                ],
                "model_states": {
                    "base": model_name,
                    "initial-sft": f"{config['initial_sft_dir']}/final",
                    **{
                        _round_name(index): path
                        for index, path in enumerate(round_states, start=1)
                    },
                },
                "output_dir": config["evaluation_dir"],
                "output_path": f"{config['evaluation_dir']}/metrics.json",
            }
        )
        results["evaluate"] = call_stage(evaluate, evaluation_config)
    return results


@app.function(**_ORCHESTRATOR_FUNCTION_OPTIONS)
def orchestrate_stamp(
    command: str,
    config: dict[str, Any],
    prepare_config: dict[str, Any] | None = None,
    round_number: int = 1,
    preference_limit: int | None = None,
) -> dict[str, Any]:
    """Run the STAMP stage graph entirely on Modal so detach survives sleep."""

    return _execute_stamp_command(
        command=command,
        config=config,
        prepare_config=prepare_config,
        round_number=round_number,
        preference_limit=preference_limit,
        call_stage=_call_stage,
    )


def _spawn_and_wait(function: Any, config: dict[str, Any]) -> dict[str, Any]:
    call = function.spawn(config)
    print(f"Spawned {function.tag} FunctionCall: {call.object_id}")
    result = call.get()
    if not isinstance(result, dict):
        raise TypeError(f"Expected stage result object, received {type(result).__name__}")
    return result


def _print_result(result: Any) -> None:
    print(json.dumps(_json_safe(_public_summary(result)), indent=2, sort_keys=True))


@app.local_entrypoint()
def main(
    command: str = "run",
    source_path: str = "work/imessages/stamp",
    sft_dir: str = "work/imessages/sft",
    run_name: str = "",
    model_name: str = DEFAULT_MODEL,
    rounds: int = DEFAULT_ROUNDS,
    candidates: int = DEFAULT_CANDIDATES,
    round_number: int = 1,
    seed: int = DEFAULT_SEED,
    limit: int = 0,
    preference_limit: int = 0,
    smoke: bool = False,
    resume: bool = True,
    force: bool = False,
) -> None:
    """Dispatch a resumable STAMP stage or the complete private pipeline.

    Commands: prepare, neutralize, classifier, sft, preferences, cpo, evaluate,
    and run. ``source_path`` must be an outgoing-only prepared split directory.

    Uploads happen locally; stage chaining runs in ``orchestrate_stamp`` so
    ``modal run --detach`` can finish SFT/CPO/eval after the laptop sleeps.

    ``preference_limit`` caps the pairs each CPO round samples candidates for.
    Unlike ``limit`` it stays inside the round config, so stages that already
    finished keep their fingerprints and are still skipped.
    """
    normalized_command = command.lower().replace("_", "-")
    aliases = {
        "upload": "prepare",
        "train-classifier": "classifier",
        "initial-sft": "sft",
        "preference": "preferences",
        "score": "preferences",
        "train-cpo": "cpo",
        "all": "run",
        "end-to-end": "run",
    }
    normalized_command = aliases.get(normalized_command, normalized_command)
    commands = {
        "prepare",
        "neutralize",
        "classifier",
        "sft",
        "preferences",
        "cpo",
        "evaluate",
        "run",
    }
    if normalized_command not in commands:
        raise ValueError(
            f"Unknown command {command!r}; choose one of {', '.join(sorted(commands))}"
        )

    if not run_name:
        run_name = datetime.now(UTC).strftime("stamp-%Y%m%d-%H%M%S")
    run_name = _safe_run_name(run_name)
    config = _base_config(
        run_name=run_name,
        model_name=model_name,
        rounds=rounds,
        candidates=candidates,
        seed=seed,
        limit=limit,
        smoke=smoke,
        resume=resume,
        force=force,
    )
    print(f"STAMP artifacts: imessage-sft-artifacts:/{run_name}/stamp")

    prepare_config: dict[str, Any] | None = None
    if normalized_command in {"prepare", "run"}:
        remote_source, inputs_are_prepared = _upload_private_inputs(source_path, run_name)
        prepare_config = dict(config)
        prepare_config.update(
            {
                "source_path": remote_source,
                "messages_path": remote_source,
                "input_path": remote_source,
                "inputs_are_prepared": inputs_are_prepared,
                "train_path": config["units_train_path"],
                "validation_path": config["units_validation_path"],
                "test_path": config["units_test_path"],
                "report_path": f"{config['prepared_dir']}/report.json",
                "judge_artifact_path": (
                    f"{config['prepared_dir']}/merge-judgments.jsonl"
                ),
                "judge_model": "gpt-5.6-luna",
                "model": "gpt-5.6-luna",
            }
        )
        if not inputs_are_prepared:
            sft_train_path, sft_validation_path = _upload_sft_splits(sft_dir, run_name)
            prepare_config.update(
                {
                    "sft_train_path": sft_train_path,
                    "sft_validation_path": sft_validation_path,
                }
            )

    call = orchestrate_stamp.spawn(
        normalized_command,
        config,
        prepare_config,
        round_number,
        preference_limit or None,
    )
    print(f"Spawned orchestrate_stamp FunctionCall: {call.object_id}")
    print(
        "Pipeline is remote; with `modal run --detach` it keeps going if this "
        "client disconnects or the laptop sleeps.",
        flush=True,
    )
    result = call.get()
    _print_result(result)
