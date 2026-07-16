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
a useful personal language model. Rewrite models use their separate pair, target-token, and
tokens-per-parameter policy.

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

## Rewrite says the model was not trained for rewrite generation

`rewrite` accepts only an artifact trained with `configs/model-rewrite-190k.yaml` or an eligible
larger rewrite preset and exported from that checkpoint. A normal reply model cannot reliably
reinterpret its input as a draft. Train and export the rewrite artifact separately, then pass it
with `--model outputs/rewrite-final`.

## Rewrite corpus stats select no model

Rewrite selection uses only the training split. It requires at least 10,000 unique training pairs,
100,000 supervised target tokens, and 1.5 target tokens per parameter. Generate and review more
unique pairs, then rebuild the splits and train a fresh 2,048-token rewrite tokenizer. Do not reuse
an old 4,096-token tokenizer or edit the selection report to bypass a failed gate.

## OpenAI pair generation fails

Put `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in the ignored project-root `.env` file. The
blind generator retries transient API and rate-limit failures, writes each successful structured
stage immediately, and can be rerun without paying to regenerate completed target identifiers. Its
report contains aggregate error types and token usage, never message text. Run
`prepare-style-targets` before `generate-convergence-pilot`.

`generate-rewrite-pairs` is intentionally blocked by default. Its target-visible, minimal-edit
neutralization produced an identity-heavy corpus and is retained only to reproduce old experiments.

An unknown model error means the configured model ID is unavailable to the account. Set
`OPENAI_MODEL` to an accessible Responses API model and rerun the same command.

## A project name, slang term, or conversational reference is interpreted incorrectly

Run `propose-entity-glossary`, inspect `review-entity-glossary`, and approve only definitions
supported by the cited private messages. Rebuild style targets afterward. The generated pair review
shows exact replies, recent turns, historical retrieval, glossary definitions, extracted
propositions, and every slang reading with its widespread/in-group/uncertain scope. Reject any group
whose interpretation is not supported there. Widespread texting slang, including short abbreviations
such as "ts" and generic address terms such as "brodie", is canonicalized directly. In-group readings
are usually proper nouns or coined names for the author's projects, people, and personal topics, and
can only be resolved through an approved glossary entry or retrieved message evidence; otherwise the
extractor must mark them uncertain and keep them verbatim. Do not solve ambiguity by approving a guessed
glossary definition or allowing future messages into retrieval.

## A rewrite pair exceeds the context window

Each neutral draft and styled target must fit together in one model context. The encoder reports the
pair identifier and token count but does not print its private text. Shorten or exclude that pair;
do not silently truncate the styled target.

## A rewrite changes the intended meaning

Use the pretrained BART adapter path, deterministic decoding, and the rewrite evaluation gate. The
legacy tiny Transformer cannot reliably preserve semantics. Promotion requires protected-fact and
local semantic checks, but those checks are not proof of equivalence; review every output before
use. A capable upstream model should remain responsible for factual content.

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

## Adapter promotion is blocked

Open the aggregate evaluation report under `work/rewrite/evaluation/`. Promotion is intentionally
blocked for changed numbers/placeholders/negation, low local semantic similarity, no measurable
style lift, empty or repeated output, or an artifact that matches a training target. Keep the
existing promoted adapter. Do not edit `ready_to_promote` manually.

If content is strong but the existing pairs show excessive exact, surface-only, or high-overlap
strata, run the local 500-pair repair pilot. It reconstructs from semantic JSON without exposing the
original wording to the second stage and keeps every pair that fails a fact check:

```bash
uv run imessage-mlx repair-rewrite-pairs-local --limit 500
```

Rebuild splits and retrain only if the pilot improves held-out content and style distributions.

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
