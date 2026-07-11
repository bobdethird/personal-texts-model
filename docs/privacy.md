# Privacy boundary

The core pipeline processes private communications entirely on the local Mac. The optional
`generate-rewrite-pairs` command is an explicit exception described below.

- The source under `~/Library/Messages` is opened read-only and is never modified.
- All processing uses a SQLite backup under `work/`, never the live database.
- Attachments are not opened or copied. Only an attachment-presence marker may be used.
- Raw handles and chat identifiers are replaced with keyed HMAC identifiers before JSONL is
  written.
- URLs, email addresses, and phone-number-shaped strings are redacted by default.
- Ordinary commands log aggregate counts, not message bodies.
- `work/`, `outputs/`, databases, arrays, and model weights are ignored by Git.
- No tokenizer, checkpoint, generated sample, or model is uploaded.
- Tests use synthetic fixtures only.
- The inference command generates terminal text and never sends a message.

Pseudonymized messages and trained weights are still sensitive and may memorize text. They are
not anonymous and must remain local.

## Optional OpenAI pair generation

`generate-rewrite-pairs` sends selected, redacted outgoing message text and opaque pair identifiers
to the configured OpenAI account so the model can produce neutral wording. It does not send raw
handles, chat identifiers, incoming messages, attachments, the Messages database, or local model
artifacts. Redaction and pseudonymization reduce exposure but do not make message content anonymous.

This command runs only when explicitly invoked with `OPENAI_API_KEY`. Review OpenAI's current data
policy and account settings before use. The resumable local output and aggregate usage report remain
under ignored private directories.
