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

All outputs contain private data and are ignored by Git.
