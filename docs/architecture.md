# Architecture

## Data boundary

The only component that touches `~/Library/Messages/chat.db` is
`src/imessage_mlx/data/snapshot.py`. It opens the source with SQLite `mode=ro`, enables
`PRAGMA query_only`, and uses SQLite's online backup API. Every later stage reads the private copy
under `work/snapshot/`.

The extractor never selects attachment paths or opens attachment files. It reads the message,
chat, handle, and join tables needed to reconstruct text ordering, then immediately replaces raw
database identities with keyed HMAC pseudonyms.

## Dataset construction

`src/imessage_mlx/data/` implements these stages:

1. `inspect_schema.py` records local table and column metadata.
2. `attributed_body.py` decodes Apple's typedstream-backed attributed strings.
3. `extract.py` filters non-text events and emits canonical pseudonymized records.
4. `normalize.py` normalizes Unicode and whitespace without correcting the author's style.
5. `redact.py` replaces obvious URLs, email addresses, and phone numbers.
6. `sessions.py` groups chronologically adjacent messages into conversations.
7. `split.py` deduplicates complete sessions and performs chronological splitting with guard bands.
8. `privacy_audit.py` verifies canonical fields, hashed identifiers, roles, and obvious-PII removal.

Splitting complete sessions before tokenizer training prevents adjacent messages or overlapping
windows from leaking across train, validation, and test data. The tokenizer sees only the training
split.

## Tokenizer

The project trains a byte-level BPE tokenizer with explicit conversation tokens:

```text
<|bos|> <|eos|> <|conversation|> <|me|> <|other|> <|turn_end|>
<|attachment|> <|url|> <|email|> <|phone|>
```

Byte-level tokenization provides complete coverage for emoji, multilingual text, unusual spelling,
and punctuation without a pretrained vocabulary.

## Model

`src/imessage_mlx/model/transformer.py` implements a decoder-only causal Transformer using MLX:

- Token embeddings with tied output weights
- Rotary positional embeddings
- Pre-normalization with RMSNorm
- Multi-head causal self-attention
- SwiGLU feed-forward blocks
- Residual connections
- Strict configured context length

The model is initialized randomly. No external weights or tokenizer are downloaded.

## Training

`src/imessage_mlx/train.py` packs the token stream into fixed causal windows and optimizes shifted
next-token cross-entropy. Training uses AdamW, gradient clipping, warmup plus cosine decay, compiled
fixed-shape MLX updates, periodic validation, best-checkpoint selection, and early stopping.

Each checkpoint contains:

```text
model.safetensors
optimizer.safetensors
random-state.safetensors
model-config.json
training-config.json
training-config.yaml
trainer-state.json
tokenizer/
```

`trainer-state.json` records the epoch, batch position, global step, trained-token count, best
validation loss, dependency versions, compilation mode, and random-initialization provenance.

## Evaluation and export

`src/imessage_mlx/evaluate.py` measures validation and untouched-test loss, overall and `me`-turn
perplexity, a unigram baseline, token n-gram overlap with training data, and obvious-PII patterns in
sampled generations. It stores aggregate counts only.

`src/imessage_mlx/export.py` creates an inference-only artifact containing model weights,
configuration, tokenizer, metrics, and split hashes. Optimizer state and source text are excluded.

`src/imessage_mlx/audit.py` rechecks the complete Gate A-D evidence, test suite, private file
permissions, Git exclusions, artifact hash, and fresh-process chat command.
