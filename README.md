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
`work/imessages/sft/`. The improved sessionizer first separates chats at six-hour
inactivity gaps, then emits one example for **every** outgoing (`me`) message. It does
not merge consecutive outgoing messages.

Each record has ordinary conversational messages and an explicit final target:

```json
{
  "format": "imessage-next-message-v1",
  "example_id": "...",
  "session_id": "...",
  "target_message_id": "...",
  "target_index": 2,
  "messages": [
    {"role": "user", "content": "Are you free later?"},
    {"role": "assistant", "content": "Probably after six"},
    {"role": "assistant", "content": "Want to grab dinner?"}
  ]
}
```

The final assistant message is the only loss target. Earlier messages—including
earlier messages from you—are context with labels set to `-100`, so no incoming
message token contributes to training loss. Each earlier outgoing message is still
trained once as the final target of its own example. Group-chat user messages retain
their pseudonymous participant as input-only metadata.

Validation is split by whole conversation session to prevent overlapping histories
from appearing in both splits. Token-length chunking keeps the latest history, keeps
the conversation marker, and, for unusually long outgoing messages, creates multiple
chunks while supervising each target token exactly once.

Use `prepare-sft --help` to change the session gap, validation fraction, or cap the
number of prior messages included in each record.

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
checkpointing, fused AdamW, and LoRA over all linear layers. Run
`uv run --extra train modal run modal_train.py --help` for all hyperparameters.
