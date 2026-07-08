# Privacy boundary

This project processes private communications entirely on the local Mac.

- The source under `~/Library/Messages` is opened read-only and is never modified.
- All processing uses a SQLite backup under `work/`, never the live database.
- Attachments are not opened or copied. Only an attachment-presence marker may be used.
- Raw handles and chat identifiers are replaced with keyed HMAC identifiers before JSONL is
  written.
- URLs, email addresses, and phone-number-shaped strings are redacted by default.
- Ordinary commands log aggregate counts, not message bodies.
- `work/`, `outputs/`, databases, arrays, and model weights are ignored by Git.
- No data, tokenizer, checkpoint, generated sample, or model is uploaded.
- Tests use synthetic fixtures only.
- The inference command generates terminal text and never sends a message.

Pseudonymized messages and trained weights are still sensitive and may memorize text. They are
not anonymous and must remain local.
