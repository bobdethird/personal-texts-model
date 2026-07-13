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
- Adapter environments disable Hugging Face telemetry, implicit credentials, experiment tracking,
  and generic do-not-track signals.
- Pretrained base weights may be downloaded from Hugging Face, but private datasets, prompts,
  predictions, adapters, and evaluation artifacts are never uploaded.
- Promoted adapters carry explicit no-upload and no-automatic-sending metadata.

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

## OpenAI convergence generation

`generate-convergence-pilot` is a second explicit hosted-processing exception. Stage A sends the
complete accepted user-written target plus an opaque target identifier to the configured OpenAI
account. Stage B is a new request with no shared response context; it receives only structured
semantic JSON and the opaque identifier. Both requests set `store=False`.

The semantic JSON is wording-blind, not anonymous. It can still contain entities, relationships,
times, emotional meaning, and other sensitive facts, so semantics, generated variants, usage logs,
manifests, predictions, and reviews remain private ignored artifacts with restrictive permissions.
Aggregate reports contain hashes, counts, token usage, and rejection classes but no message text or
API key. `store=False` disables response storage for the request; it does not supersede the
provider's account-level processing or retention terms.

Training, local semantic validation, adapter inference, convergence evaluation, and promotion remain
on the Mac. No command sends a generated message automatically.

## Local targeted repair

`repair-rewrite-pairs-local` is not a hosted fallback. Both semantic extraction and blinded
reconstruction run through the local MLX-LM environment. Stage B receives structured semantic JSON,
not the original personal wording. Deterministic fact checks reject changed numbers, placeholders,
or negation, and rejected records retain their existing neutral draft. The aggregate report contains
pair identifiers and counts but no original or generated message text.
