---
stepsCompleted: [1]
inputDocuments:
  - README.md
  - pyproject.toml
  - configs/adapter-bart-base.yaml
  - src/imessage_mlx/adapter_worker.py
workflowType: research
lastStep: 2
research_type: technical
research_topic: compact encoder-decoder alternatives to BART for personal text rewriting
research_goals: compare quality, size, Apple Silicon compatibility, and LoRA feasibility
date: 2026-07-14
web_research_enabled: true
source_verification: true
---

# Compact Seq2Seq Architecture Research

## Technical Research Scope Confirmation

**Research topic:** Compact alternatives to the project's BART rewrite adapter, including
MarianMT/OPUS-MT, NLLB, mBART, T5/Flan-T5, and task-relevant alternatives.

**Research goals:**

- Compare checkpoint size, memory, tokenizer behavior, intended task, and licensing.
- Determine whether each candidate can use LoRA on the existing private source/target pairs.
- Assess compatibility with the current PyTorch/Transformers/PEFT worker and Apple Silicon.
- Rank candidates for controlled experiments without weakening the existing evaluation gates.

**Method:** Current model cards, official documentation, published papers, and repository
implementation evidence were cross-checked. Checkpoint-level facts are distinguished from
architecture-family claims.

## Technology Stack Analysis

### Current Project Baseline

The promoted rewrite path is `facebook/bart-base`, a 139.4M-parameter English encoder-decoder
checkpoint, trained through PyTorch, Transformers, and PEFT on MPS. MLX is not used for BART
training or inference. The current rank-8 LoRA adapter targets `q_proj` and `v_proj` across the
encoder, decoder self-attention, and decoder cross-attention. Its input contract is a bare neutral
draft mapped to a styled target, with independent 256-token limits.

The baseline is unusually important because it already passed this project's semantic,
protected-fact, style, fluency, and memorization gates. A smaller parameter count is not by itself
evidence of a better model.

Sources:

- [BART model card](https://huggingface.co/facebook/bart-base)
- [BART Transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/bart)
- [BART paper](https://aclanthology.org/2020.acl-main.703/)
- Local implementation: `configs/adapter-bart-base.yaml`,
  `src/imessage_mlx/adapter_worker.py`

### Candidate Families and Checkpoints

#### T5 and Flan-T5

`google-t5/t5-small` has about 60.5M parameters. `google/flan-t5-small` has about 77.0M and
`google/flan-t5-base` about 247.6M. All use a text-to-text encoder-decoder architecture and a
32K SentencePiece vocabulary. Flan-T5 adds instruction tuning, making it a more natural candidate
for an explicit task prefix such as `rewrite in my texting style:`.

This is the strongest compact family to benchmark. Flan-T5-small is roughly 45% of BART-base's
parameter count and is Apache-2.0 licensed. It is not a drop-in adapter replacement: its attention
modules are named `q` and `v`, its tokenizer differs, and a task prefix should be applied
consistently during training and inference.

Sources:

- [T5-small model card](https://huggingface.co/google-t5/t5-small)
- [Flan-T5-small model card](https://huggingface.co/google/flan-t5-small)
- [Flan-T5-base model card](https://huggingface.co/google/flan-t5-base)
- [T5 paper](https://jmlr.org/papers/v21/20-074.html)
- [Flan paper](https://jmlr.org/papers/volume25/23-0870/23-0870.pdf)

#### MarianMT and OPUS-MT

Marian is the translation architecture/runtime; OPUS-MT is a collection of separately trained
Marian checkpoints. A typical small checkpoint such as `Helsinki-NLP/opus-mt-en-de` is around
74M parameters, uses SentencePiece, and is trained for one translation direction.

Its size is attractive and its module structure is close to BART, so PEFT LoRA is mechanically
feasible with explicit `q_proj` and `v_proj` targets. However, English-to-German or other
directional translation pretraining is a poor semantic prior for English-to-English style
rewriting. There is no canonical small OPUS checkpoint pretrained specifically for open-ended
same-language rewriting. Marian should therefore be treated as a diagnostic experiment, not the
leading replacement.

Sources:

- [MarianMT documentation](https://huggingface.co/docs/transformers/main/en/model_doc/marian)
- [OPUS-MT repository](https://github.com/Helsinki-NLP/Opus-MT)
- [Representative OPUS-MT checkpoint](https://huggingface.co/Helsinki-NLP/opus-mt-en-de)

#### NLLB-200

`facebook/nllb-200-distilled-600M` is approximately 600–615M parameters with a 256K vocabulary and
explicit source/target language codes. It is designed for translation across 200 languages, not
same-language personal rewriting. Its model card uses CC-BY-NC-4.0, labels production use
out-of-scope, and warns that training inputs did not exceed 512 tokens.

LoRA is technically feasible, but it does not reduce the frozen base model's inference footprint.
NLLB is more than four times BART-base's parameter count, introduces language-token handling, and
has a restrictive license. It should be excluded unless broad multilingual translation becomes a
core requirement.

Sources:

- [NLLB-200 distilled model card](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [NLLB Transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/nllb)
- [NLLB paper](https://arxiv.org/abs/2205.12654)

#### mBART-50

`facebook/mbart-large-50-many-to-many-mmt` has about 611M parameters, a 250K SentencePiece
vocabulary, and mandatory language-control tokens. It combines multilingual denoising pretraining
with translation fine-tuning, but the commonly available checkpoint is substantially larger than
BART-base and translation-specialized. It also lacks a clear checkpoint-specific license
declaration in current metadata.

It remains more relevant than NLLB if multilingual denoising initialization is desired, but it is
not a compact replacement.

Sources:

- [mBART-50 model card](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)
- [mBART Transformers documentation](https://huggingface.co/docs/transformers/model_doc/mbart)
- [mBART paper](https://aclanthology.org/2020.tacl-1.47/)

#### BlenderBot Small

`facebook/blenderbot_small-90M` is an approximately 90M-parameter English dialogue model and is
more task-aligned to response generation than any translation checkpoint. It is Apache-2.0 and
worth including as a dialogue-specific baseline. Its tokenizer lowercases text, however, which can
erase capitalization signals that matter to personal style reconstruction. Its older
open-domain-dialogue training is also less aligned to semantic-preserving rewriting than BART or
Flan-T5.

Sources:

- [BlenderBot Small model card](https://huggingface.co/facebook/blenderbot_small-90M)
- [BlenderBot Small documentation](https://huggingface.co/docs/transformers/model_doc/blenderbot-small)
- [BlenderBot paper](https://arxiv.org/abs/2004.13637)

### LoRA and Apple Silicon Tooling

Hugging Face PEFT supports sequence-to-sequence LoRA through
`TaskType.SEQ_2_SEQ_LM`. BART and T5 have built-in target-module mappings. Marian, mBART, and
NLLB/M2M100 can still be adapted by declaring their target modules explicitly.

Recommended initial attention targets are:

- BART, Marian, mBART, and NLLB/M2M100: `q_proj`, `v_proj`.
- T5 and Flan-T5: `q`, `v`.

The practical local route remains PyTorch plus PEFT on MPS. `mlx-lm` is designed around
decoder-only language models and cannot directly train these encoder-decoder families. Apple
provides a standalone T5 inference example in `mlx-examples`, but no generic seq2seq LoRA trainer.
A native MLX implementation would require a custom encoder-decoder training loop, cross-attention
adapter injection, checkpoint conversion, generation, and parity tests.

Standard bitsandbytes NF4 QLoRA is not a supported Apple-MPS path. MLX can train adapters over its
own quantized linear layers, but doing so here would be a custom quantized-LoRA implementation,
not a configuration change to the current worker.

Sources:

- [PEFT LoRA API](https://huggingface.co/docs/peft/en/package_reference/lora)
- [PEFT quantization guide](https://huggingface.co/docs/peft/en/developer_guides/quantization)
- [MLX T5 example](https://github.com/ml-explore/mlx-examples/tree/main/t5)
- [MLX LoRA scope discussion](https://github.com/ml-explore/mlx-examples/issues/560)
- [mlx-lm LoRA documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

### Preliminary Technology Ranking

1. Keep `facebook/bart-base` as the control and promoted model.
2. Benchmark `google/flan-t5-small` as the smallest credible replacement.
3. Benchmark `google/flan-t5-base` only if Flan-T5-small shows a useful quality signal but lacks
   capacity.
4. Optionally test BlenderBot Small as a dialogue prior, with capitalization loss explicitly
   measured.
5. Test one small Marian/OPUS-MT checkpoint only as an architecture/pretraining ablation.
6. Do not spend a full training run on NLLB or mBART for the current English rewrite objective.

### Confidence and Open Questions

Confidence is high for model sizes, task intent, tokenizer requirements, and PEFT feasibility.
Exact peak memory and throughput cannot be inferred from parameter count and must be measured on
the target Mac with the project's sequence lengths and batch settings. The decisive uncertainty is
empirical: whether Flan-T5-small's instruction prior can preserve facts and personal style as well
as the existing BART adapter.
