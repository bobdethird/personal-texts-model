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
100,000 supervised target tokens, and two target tokens per parameter. Generate and review more
unique pairs, then rebuild the splits and train a fresh 2,048-token rewrite tokenizer. Do not reuse
an old 4,096-token tokenizer or edit the selection report to bypass a failed gate.

## OpenAI pair generation fails

Put `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in the ignored project-root `.env` file. The
generator retries transient API and rate-limit failures, writes each successful structured batch
immediately, and can be rerun without paying to regenerate completed pair identifiers. Its report
contains aggregate error types and token usage, never message text.

An unknown model error means the configured model ID is unavailable to the account. Set
`OPENAI_MODEL` to an accessible Responses API model and rerun the same command.

## A rewrite pair exceeds the context window

Each neutral draft and styled target must fit together in one model context. The encoder reports the
pair identifier and token count but does not print its private text. Shorten or exclude that pair;
do not silently truncate the styled target.

## A rewrite changes the intended meaning

The rewrite model is a tiny Transformer trained from scratch and semantic equivalence is not
guaranteed. Lower temperature may reduce variation, but every rewrite must still be reviewed before
use. A capable external model should remain responsible for the factual content.

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
