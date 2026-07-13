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
9. `rewrite.py` validates neutral-to-styled pairs and serializes the separate rewrite task.
10. `pair_generation.py` optionally asks OpenAI to neutralize recent outgoing messages in resumable,
    structured batches without logging message text.
11. `adapters.py` converts chronological pairs into BART source/target and MLX-LM
    prompt/completion records, removes normalized and fuzzy cross-split duplicates, reports signal
    strata, filters deterministic fact conflicts and length outliers, caps exact/surface-only
    examples, and writes a frozen architecture-benchmark subset.
12. `repair.py` can reconstruct only low-signal pairs through blinded semantic JSON. Failed fact,
    schema, or content checks retain the original pair.

Splitting complete sessions before tokenizer training prevents adjacent messages or overlapping
windows from leaking across train, validation, and test data. The tokenizer sees only the training
split.

## Tokenizer

The project trains a byte-level BPE tokenizer with explicit conversation tokens:

```text
<|bos|> <|eos|> <|conversation|> <|me|> <|other|> <|turn_end|>
<|attachment|> <|url|> <|email|> <|phone|> <|rewrite|> <|draft|>
```

Byte-level tokenization provides complete coverage for emoji, multilingual text, unusual spelling,
and punctuation without a pretrained vocabulary. The trainer reserves the complete byte alphabet.
Reply models request a 4,096-token vocabulary; the smaller rewrite models request 2,048.

## Model

`src/imessage_mlx/model/transformer.py` implements a decoder-only causal Transformer using MLX:

- Token embeddings with tied output weights
- Rotary positional embeddings
- Pre-normalization with RMSNorm
- Multi-head causal self-attention
- SwiGLU feed-forward blocks
- Residual connections
- Strict configured context length

The reply model and legacy tiny rewrite baseline are initialized randomly. The recommended rewrite
path instead trains a low-rank adapter over a checksum-cached pretrained base model. Base weights
remain separate from the sensitive local adapter.

## Training

`src/imessage_mlx/train.py` packs the token stream into fixed causal windows and optimizes shifted
next-token cross-entropy. Training uses AdamW, gradient clipping, warmup plus cosine decay, compiled
fixed-shape MLX updates, periodic validation, best-checkpoint selection, and early stopping.

Rewrite data uses an explicit task prompt:

```text
<|bos|><|rewrite|>
<|draft|>neutral draft<|turn_end|>
<|me|>styled target<|turn_end|>
<|eos|>
```

Complete pairs are packed without crossing a context boundary. An aligned loss mask supervises only
the styled target and its turn terminator; prompt and padding tokens do not affect optimization.
Reply models retain the original unmasked causal objective. Rewrite and reply models are exported as
separate artifacts.

The adapter path benchmarks two isolated environments against the same frozen records:

- BART-base uses PEFT `SEQ_2_SEQ_LM` LoRA on MPS. The encoder receives the complete neutral draft
  and the decoder learns only the styled target.
- Qwen3-0.6B-4bit uses MLX-LM QLoRA with `--mask-prompt`, so instruction and draft tokens do not
  contribute to loss.

The environments live under ignored `work/envs/` and do not change the pinned custom-model
environment. Training uses deterministic seeds, small batches, gradient accumulation, validation,
best-adapter saves, and early stopping. BART applies an exact 256-token source/target guard before
training; MLX-LM receives the same maximum sequence length. Adapters are neither fused nor uploaded.

Multi-register convergence is a separate v1 data contract layered onto the accepted BART splits.
Targets are selected deterministically within their existing train/validation/test assignment.
OpenAI Stage A extracts typed semantics from the target; a separate Stage B request receives only
the semantic object and emits exactly four labeled sources. The manifest fingerprints the input
files, model, prompts, schemas, and style set. Semantics and variants checkpoint independently, while
only complete target groups are published.

Each generated row carries `target_id`, `source_pair_id`, `variant_kind`, and `split`. Adapter
deduplication treats repeated targets from the same owner group as intentional but rejects matches
owned by another group or split. Local sentence-transformer scoring runs before dataset assembly.
The pilot uses all four rows and reports effective target exposure; larger runs must reconsider a
group-aware rotating sampler if memorization rises.

Model selection is task-specific. Reply selection retains its original one-million-token hard
minimum and ten-token-per-parameter heuristic. Rewrite selection considers only the training split,
requires at least 10,000 unique pairs and 100,000 supervised target tokens, then selects the largest
rewrite candidate with at least 1.5 supervised target tokens per parameter. It returns no model
when the gates fail. The rewrite presets are approximately 188K and 291K parameters at a full
2,048-token vocabulary.

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
sampled generations for reply models. `src/imessage_mlx/rewrite_evaluation.py` separately evaluates
rewrite predictions with local embedding similarity, protected-fact preservation, a held-out
character-style classifier, style-marker gap closure, malformed/repetition rates, and exact
training-target matches. Reports store aggregate values, never matched text.

`src/imessage_mlx/convergence_evaluation.py` validates complete four-register groups, then reports
protected facts, target/source similarity, normalized input copying, style distance and variance,
within-target output agreement, worst-register behavior, fluency, and training-target memorization.
Baseline and pilot reports must have the same challenge fingerprint. Expansion uses explicit gates
rather than a weighted score and still requires a private grouped human review.

`src/imessage_mlx/export.py` creates an inference-only artifact containing model weights,
configuration, tokenizer, task capabilities, metrics, and split hashes. Optimizer state and source
text are excluded. Adapter export additionally records pretrained-base provenance, adapter hashes,
model license, deterministic generation defaults, no-upload/no-auto-send warnings, and a
timestamped rollback artifact when replacing a prior promotion.

`src/imessage_mlx/audit.py` rechecks the complete Gate A-D evidence, test suite, private file
permissions, Git exclusions, artifact hash, and fresh-process chat command.
