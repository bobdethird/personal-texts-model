# iMessage Downloader

A small macOS-only command-line tool that makes a read-only SQLite snapshot of
`~/Library/Messages/chat.db` and exports message text to private JSONL and CSV files.

## Setup

```bash
uv sync
```

Grant Full Disk Access to Cursor or the terminal that runs the command. Contact names are
optional and require separate Contacts permission.

## Download

```bash
uv run imessage-download doctor
uv run imessage-download download
```

By default, files are written to `work/imessages/` with owner-only permissions:

- `chat.db`: consistent SQLite snapshot
- `messages.jsonl`: structured message records
- `messages.csv`: spreadsheet-friendly export
- `schema.json`: source schema report
- `extraction-report.json`: row accounting and recovery statistics
- `pseudonym-key`: local key used to pseudonymize identifiers

Use `--include-contact-names` to resolve senders from macOS Contacts. Use
`--redact-private-data` to replace URLs, email addresses, and phone numbers in message text.
Run `uv run imessage-download download --help` for all options.

The structured exports omit deleted or retracted rows, reactions, system events, and
attachment-only rows. The `chat.db` snapshot retains the complete database.

## Conversation chunking

After downloading, group messages from the same chat into conversation sessions:

```bash
uv run imessage-download chunk
```

This command is separate from `download`; downloading does not run it automatically. By
default, it writes `work/imessages/sessions.jsonl` and
`work/imessages/session-report.json`. A new session starts after six hours of inactivity,
and consecutive messages from the same sender within two minutes are merged into one turn.
Use `uv run imessage-download chunk --help` to change the paths or timing windows.

All outputs contain private data and are ignored by Git.

## Conversational SFT data

Create next-message training data from the message export:

```bash
uv run imessage-download prepare-sft
```

This writes private `train.jsonl`, `validation.jsonl`, and `report.json` files under
`work/imessages/sft/`. The sessionizer first separates chats at six-hour inactivity
gaps, then emits **one example per conversation session** that contains at least one
outgoing (`me`) message. Consecutive outgoing messages are not merged.

Each record is a full multi-turn transcript:

```json
{
  "format": "imessage-session-v2",
  "example_id": "...",
  "session_id": "...",
  "supervised_indexes": [1, 2],
  "messages": [
    {"role": "user", "content": "Are you free later?"},
    {"role": "assistant", "content": "Probably after six"},
    {"role": "assistant", "content": "Want to grab dinner?"}
  ]
}
```

Every assistant turn is a loss target. User turns stay in the transcript as context
with labels set to `-100`, so no incoming message token contributes to training loss.
Because the model is causal, supervising all of your turns in one session gives each
turn the same conditional history as emitting a separate example per message, without
re-encoding the shared prefix over and over. Group-chat user messages retain a short
pseudonymous participant prefix as input-only metadata.

Export placeholders such as `<|attachment|>` are stripped during preparation and
messages that become empty are dropped, so the model never learns to emit them.
Transcripts are rendered with the base model's own `<|im_start|>`/`<|im_end|>` chat
tokens (plus a fixed system header); no new vocabulary is added, which keeps every
delimiter a well-trained token and generation stops reliable.

Validation is split by whole conversation session to prevent overlapping histories
from appearing in both splits. During training, sessions longer than the model context
are split into windows (default `max_length=4096`). Each of your messages is fully
contained in one window and supervised there exactly once; earlier turns may be
repeated as masked context so later replies still see recent history.

Use `prepare-sft --help` to change the session gap or validation fraction.

## Retrieval personalization baseline

Before training an adapter, build a training-free baseline that retrieves similar
incoming messages and shows the base model how you replied. The pair dataset is
derived from the existing whole-session split, so validation conversations never
enter the retrieval index. Session SFT files remain unchanged.

Install the local embedding dependencies, derive pairs, and build the private index:

```bash
uv sync --extra personalize
uv run imessage-download prepare-pairs
uv run --extra personalize imessage-download build-retrieval-index
```

`prepare-pairs` writes `imessage-pair-v1` files under `work/imessages/pairs/`.
Each pair keeps the last incoming message, your reply, and the full preceding
session context. `build-retrieval-index` embeds only training queries with
`sentence-transformers/all-MiniLM-L6-v2`, then writes normalized vectors and
slim pair metadata under `work/imessages/retrieval/`. Both directories contain
private text and are ignored by Git.

Induce a short style card from the training replies with one model call. Running it
on Modal avoids loading the 4B model locally:

```bash
uv run --extra train modal run modal_train.py::style_card_main \
  --run-name personal-qwen
```

Then compare the same held-out targets with the base prompt, retrieval demonstrations,
and retrieval plus the style card:

```bash
uv run --extra train modal run modal_train.py::personalize_main \
  --run-name personal-qwen
```

The comparison is saved to the `imessage-sft-artifacts` Volume at
`/personal-qwen/personalization/sample-examples.json`. Each example includes the
gold reply, all three generations, retrieved pair IDs and scores, and embedding
cosine to the gold reply. That cosine uses the semantic retrieval encoder, so it
is a content proxy rather than an authorship-verification score.

After downloading the JSON, render a private browser-friendly report with:

```bash
uv run imessage-download render-personalization-report
```

The default output is
`outputs/personal-qwen/personalization/report.html`. It presents conversation
bubbles, your real reply, the three generated alternatives, summary scores, and
collapsed retrieval details.

For local style-card induction instead, run:

```bash
uv run --extra personalize imessage-download induce-style-card
```

Pass the resulting file to Modal sampling with
`--style-card-path work/imessages/style-card.md`.

## STAMP-style iMessage rewriting

This is a practical adaptation of
[ISI-NLP STAMP](https://github.com/isi-nlp/STAMP), not a byte-for-byte reproduction of
its Llama-2/ParaNMT setup. The experiment is separate from next-message prediction and learns
to rewrite a content-equivalent neutral draft in the phone owner's texting style. It uses only
outgoing messages; incoming messages and conversation context are never uploaded as style data.

Install the experiment dependencies and prepare leakage-safe style units:

```bash
uv sync --extra stamp
export OPENAI_API_KEY=...
uv run --extra stamp imessage-download prepare-stamp
```

Preparation first groups only adjacent outgoing bubbles from the same chat and session that
arrived within two minutes with no intervening incoming message. The configured merge judge
(default `gpt-5.6-luna`) then decides whether each candidate chain is one continuing thought
or separate sends. Failed judge calls leave bubbles separate. Merged targets retain newlines,
so the experiment can learn multi-bubble rhythm. Reports contain counts and hashes, never
message text.

The resulting outgoing-only style units live under `work/imessages/stamp/`. Modal uploads only
that directory, asks Qwen to combine each unit into one fluent neutral draft, and stores the
private neutral/original pairs in the artifact Volume. Run a minimal end-to-end check before a
production experiment:

```bash
uv run --extra stamp modal run modal_stamp.py --run-name personal-stamp --smoke
uv run --extra stamp modal run --detach modal_stamp.py --run-name personal-stamp
```

The production pipeline trains the original-vs-neutral style classifier, initial
neutral-to-personal LoRA, then runs three rounds of STAMP-style hope/fear candidate generation
and reference-free CPO on one `A100-40GB`. Checkpoints and evaluation JSON are stored privately under
`/personal-stamp/stamp/` in the `imessage-sft-artifacts` Modal Volume.
Modern TRL no longer ships `CPOTrainer`, while its old CPO release predates Qwen 3.
The pipeline therefore implements the same reference-free sigmoid CPO objective directly on
Transformers and uses current TRL only for initial SFT.

After downloading the evaluation JSON, render the private comparison report:

```bash
uv run --extra stamp modal volume get imessage-sft-artifacts \
  /personal-stamp/stamp/evaluation/evaluation.json \
  outputs/personal-stamp-evaluation.json
uv run imessage-download render-stamp-report \
  --results outputs/personal-stamp-evaluation.json \
  --output outputs/personal-stamp-report.html
```

Automatic style probability is an in-domain classifier proxy, not proof that a person authored
the output. Semantic similarity, normalized base-model likelihood, length diagnostics, and
structural-anchoring measurements should be interpreted together with the example report.

## Train on Modal (A100)

Install the optional training tools and authenticate Modal:

```bash
uv sync --extra train
uv run --extra train modal setup
```

Prepare the SFT files locally, then launch LoRA training on one 80 GB A100:

```bash
uv run imessage-download prepare-sft
uv run --extra train modal run --detach modal_train.py
```

The local entrypoint uploads only the prepared SFT directory to the private
`imessage-sft-data` Modal Volume. Checkpoints, the final adapter, tokenizer, and
training summary are committed to the `imessage-sft-artifacts` Volume. Model downloads
are cached separately in `imessage-sft-model-cache`.

Common overrides:

```bash
uv run --extra train modal run --detach modal_train.py \
  --run-name personal-qwen \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --max-length 4096 \
  --epochs 3
```

Resume the latest checkpoint in the same output directory with the same run name:

```bash
uv run --extra train modal run --detach modal_train.py \
  --run-name personal-qwen --resume
```

The Modal function is pinned to `gpu="A100-80GB"` and uses BF16, TF32, gradient
checkpointing, fused AdamW, and LoRA over all linear layers. The default context
length is 4096 tokens. Run `uv run --extra train modal run modal_train.py --help`
for all hyperparameters.
