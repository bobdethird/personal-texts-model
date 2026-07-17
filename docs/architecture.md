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
6. `sessions.py` groups chronologically adjacent messages into conversations for the reply model.
7. `split.py` deduplicates complete sessions and performs chronological splitting with guard bands.
8. `privacy_audit.py` verifies canonical fields, hashed identifiers, roles, and obvious-PII removal.
9. `llm_dataset.py` generates judge-screened rewrite pairs with hosted model decisions only.
10. `censor.py` screens every accepted pair and is the only publisher of training splits.

Splitting complete sessions before tokenizer training prevents adjacent messages or overlapping
windows from leaking across train, validation, and test data. The tokenizer sees only the training
split.

## LLM-only rewrite dataset

`llm_dataset.py` deliberately contains no linguistic or quality heuristics. Local code performs
only mechanical work: it slices each chat's chronological messages into fixed-size windows, labels
speakers with first names, checkpoints finished windows for resume, and serializes results. Every
data decision is made by a hosted GPT model through structured-output calls:

1. A proposal call reads the window and decides how the owner's messages group into turns, which
   turns are usable, and what a generic assistant would have drafted to express the same meaning
   (slang translated from the model's own knowledge and the conversation, proper nouns preserved
   verbatim). It also writes a one-sentence context note for human review.
2. A judge call re-reads the window plus the proposed pairs and accepts or rejects each one,
   checking meaning preservation, assistant-neutral drafting, verbatim proper nouns, and coherent
   turn grouping.
3. A censor call (in `censor.py`) screens each accepted pair in full — draft, original, and note —
   and excludes rows containing adult content (vulgar, violent, or sexual material) or secrets
   and super-personal information (passwords, API keys, credentials, verification codes,
   financial numbers, government identifiers). Exclusion drops the row entirely; nothing is
   redacted or rewritten. Screening failures are excluded too, so publication is fail-closed.

Responses are validated mechanically only: echoed identifiers, in-range strictly ascending
owner-only indices, no index reuse, one verdict per candidate or row, and category consistency.
Malformed responses are retried with the previous error attached; windows or batches that stay
malformed are recorded as failed with no local fallback. Censor-allowed pairs are ordered
chronologically and sliced into train/valid/test by configurable fractions, then serialized as
generic records, BART `source`/`target` files, and MLX-LM `prompt`/`completion` files under one
dataset root. Only censor-approved pairs ever reach those files.

## Tokenizer

The project trains a byte-level BPE tokenizer with explicit conversation tokens:

```text
<|bos|> <|eos|> <|conversation|> <|me|> <|other|> <|turn_end|>
<|attachment|> <|url|> <|email|> <|phone|>
```

Byte-level tokenization provides complete coverage for emoji, multilingual text, unusual spelling,
and punctuation without a pretrained vocabulary. The trainer reserves the complete byte alphabet.
Reply models request a 4,096-token vocabulary.

## Model

`src/imessage_mlx/model/transformer.py` implements a decoder-only causal Transformer using MLX:

- Token embeddings with tied output weights
- Rotary positional embeddings
- Pre-normalization with RMSNorm
- Multi-head causal self-attention
- SwiGLU feed-forward blocks
- Residual connections
- Strict configured context length

The reply model is initialized randomly. The rewrite path instead trains a low-rank adapter over a
checksum-cached pretrained base model. Base weights remain separate from the sensitive local
adapter.

## Training

`src/imessage_mlx/train.py` packs the token stream into fixed causal windows and optimizes shifted
next-token cross-entropy. Training uses AdamW, gradient clipping, warmup plus cosine decay, compiled
fixed-shape MLX updates, periodic validation, best-checkpoint selection, and early stopping.

The adapter path trains isolated environments against the same frozen records:

- BART-base (and the Flan-T5/Marian challengers) use PEFT `SEQ_2_SEQ_LM` LoRA on MPS. The encoder
  receives the complete draft and the decoder learns only the owner's real text.
- Qwen3-0.6B-4bit uses MLX-LM QLoRA with `--mask-prompt`, so instruction and draft tokens do not
  contribute to loss.

The environments live under ignored `work/envs/` and do not change the pinned custom-model
environment. Training uses deterministic seeds, small batches, gradient accumulation, validation,
best-adapter saves, and early stopping. Adapters are neither fused nor uploaded.

Reply model selection retains its one-million-token hard minimum and ten-token-per-parameter
policy.

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
sampled generations for reply models. Reports store aggregate values, never matched text.

Rewrite adapters are reviewed by a human: `review-llm-dataset` renders generated pairs and
`review-censor` renders every censor-excluded row before training, and `predict-adapter` produces
held-out predictions to read directly. There is no automated promotion gate; nothing replaces human
judgment of the private outputs.

`src/imessage_mlx/export.py` creates an inference-only reply artifact containing model weights,
configuration, tokenizer, task capabilities, metrics, and split hashes. Optimizer state and source
text are excluded.

`src/imessage_mlx/audit.py` rechecks the complete Gate A-D evidence, test suite, private file
permissions, Git exclusions, artifact hash, and fresh-process chat command.
