# Troubleshooting

## `authorization denied` when opening `chat.db`

Grant Full Disk Access to the exact application running the command:

```text
System Settings → Privacy & Security → Full Disk Access
```

Quit and restart the application completely, then run:

```bash
uv run imessage-mlx doctor
```

Do not loosen the permissions of `~/Library/Messages` or train from the live database.

## The snapshot reports that the source changed

Messages received during the backup can change the live database. Wait briefly and rerun the
snapshot command. SQLite's backup API keeps the copy consistent; the extra check prevents an
ambiguous run from continuing.

## `prepare` stops on body recovery

Do not lower the recovery threshold immediately. Update to the latest code and rerun. Messages often
stores visible text in Apple's typedstream `attributedBody` rather than the ordinary `text` column.
The pipeline deliberately stops when too many eligible messages cannot be decoded.

If generated output contains strings such as `__kIMMessagePartAttributeName`, the dataset was built
with an incorrect attributed-body decoder. Delete the affected private derived artifacts, update the
code, rerun `prepare`, and confirm that the metadata scan and privacy audit report zero failures
before retraining.

## The reply corpus has fewer than one million tokens

The reply task refuses a normal run because a from-scratch Transformer would mostly memorize the
corpus. You can still inspect the synthetic smoke model in the test suite, but do not present it as
a useful personal language model.

## The model repeats phrases or produces incoherent text

This is expected for very small from-scratch models. Try conservative generation settings:

```bash
uv run imessage-mlx chat \
  --model outputs/final \
  --temperature 0.5 \
  --top-p 0.8 \
  --max-new-tokens 24 \
  --repetition-penalty 1.2
```

Sampling changes cannot add factual knowledge or reasoning. A larger model trained on the same small
corpus may memorize more without becoming more coherent.

## OpenAI dataset generation fails

Put `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in the ignored project-root `.env` file.
`build-llm-dataset` retries transient API and rate-limit failures, checkpoints every completed
window into `results.jsonl`, and can be rerun on the same output directory without paying to
regenerate finished windows. The censor checkpoints per pair into `censor.jsonl` the same way.
Reports contain aggregate error types and token usage, never message text.

An unknown model error means the configured model ID is unavailable to the account. Set
`OPENAI_MODEL` to an accessible Responses API model and rerun the same command. Changing the model,
window size, or extracted messages requires a fresh output directory because the resume checks
refuse mixed artifacts.

## A generated pair misreads slang or a project name

There is no local dictionary to edit; interpretation is entirely the hosted model's job. Read
`review-llm-dataset` output — every pair carries the model's context note and the judge's verdict
next to its timestamp. If misreadings cluster, rerun the pilot with a stronger `--model` (or a
stronger `--judge-model`) and compare acceptance rates before scaling up.

## The censor excluded too much or too little

Read `review-censor` output: it renders every excluded row with its category and the censor's
reason. The censor intentionally errs toward exclusion when uncertain, and screening failures are
excluded rather than published. If exclusions look wrong in bulk, rerun `censor-llm-dataset` with a
stronger `--model` on a fresh output directory; do not hand-edit the published splits.

## A rewrite changes the intended meaning

Use deterministic decoding and read the held-out predictions directly before trusting an adapter.
Dataset acceptance is judged per pair during generation, but that is not proof of equivalence;
review every output before use. A capable upstream model should remain responsible for factual
content.

## An adapter environment is missing a package

The PEFT/MPS and MLX-LM stacks are deliberately isolated from the main environment. Recreate only
the affected environment:

```bash
uv run imessage-mlx setup-adapter-environment bart --output work/envs/bart
uv run imessage-mlx setup-adapter-environment qwen --output work/envs/mlx-lm
```

Do not install either stack into `.venv` or remove version bounds from `pyproject.toml`.

## `ModuleNotFoundError: No module named 'imessage_mlx'` after `uv run`

Some macOS `Documents` volumes mark editable-install `.pth` files as hidden, so Python skips the
project source path. Use a non-editable main install and keep that mode enabled for `uv run`:

```bash
export UV_NO_EDITABLE=1
uv sync --no-editable
uv run imessage-mlx rewrite-adapter "your draft"
```

The isolated BART and Qwen environments are unaffected. Re-run the sync after changing project
source because a non-editable install copies the package into `.venv`.

## The Mac becomes hot or runs out of memory

Run only one training process at a time. The supplied settings use batch size 1–2, gradient
accumulation, and 256-token limits. The measured BART-base peak was about 3.05 GB on the 24 GB M4.
Close other GPU-heavy applications before training; do not increase batch size merely to shorten
the run.

## Training was interrupted

Resume from the atomic `last` checkpoint using the same model configuration, tokenizer, and token
arrays:

```bash
uv run imessage-mlx train \
  --config configs/model-1m.yaml \
  --data work/tokens \
  --tokenizer outputs/tokenizer \
  --output outputs/runs/my-model \
  --resume-from outputs/runs/my-model/last
```

Resume requires the exact tokenizer copied into the checkpoint. The command stops if a tokenizer
was retrained or replaced, even when its vocabulary size is unchanged.

## The privacy audit fails

Do not continue to training. The report contains aggregate failure counts without message text. Fix
the extractor or redaction rule, rebuild the private dataset from the unchanged snapshot, and rerun:

```bash
uv run imessage-mlx privacy-audit
```

Only continue when `passed` is `true`.
