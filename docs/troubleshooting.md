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

## The corpus has fewer than one million tokens

The project refuses a normal run because a from-scratch Transformer would mostly memorize the
corpus. You can still inspect the synthetic smoke model in the test suite, but do not present it as a
useful personal language model.

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

## The privacy audit fails

Do not continue to training. The report contains aggregate failure counts without message text. Fix
the extractor or redaction rule, rebuild the private dataset from the unchanged snapshot, and rerun:

```bash
uv run imessage-mlx privacy-audit
```

Only continue when `passed` is `true`.
