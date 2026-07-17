# Privacy boundary

The core pipeline processes private communications entirely on the local Mac. The optional
`build-llm-dataset` and `censor-llm-dataset` commands are the explicit hosted-processing
exceptions described below.

- The source under `~/Library/Messages` is opened read-only and is never modified.
- All processing uses a SQLite backup under `work/`, never the live database.
- Attachments are not opened or copied. Only an attachment-presence marker may be used.
- Raw handles and chat identifiers are replaced with keyed HMAC identifiers before JSONL is
  written. When `include_contact_names` is enabled, the extracted JSONL also contains local
  Contacts display names in `sender_name`; phone numbers and email handles remain omitted.
- URLs, email addresses, and phone-number-shaped strings are redacted by default.
- Ordinary commands log aggregate counts, not message bodies.
- `work/`, `outputs/`, databases, arrays, and model weights are ignored by Git.
- No tokenizer, checkpoint, generated sample, or model is uploaded.
- Tests use synthetic fixtures only.
- The inference commands generate terminal text and never send a message.
- Adapter environments disable Hugging Face telemetry, implicit credentials, experiment tracking,
  and generic do-not-track signals.
- Pretrained base weights may be downloaded from Hugging Face, but private datasets, prompts,
  predictions, adapters, and evaluation artifacts are never uploaded.

Contact names, pseudonymized messages, and trained weights are sensitive and may reveal or memorize
identity and message content. The dataset is not anonymous and must remain local.

## OpenAI dataset generation

`build-llm-dataset` sends complete conversation windows — redacted incoming and outgoing message
text, first-name speaker labels, and local timestamps — to the configured OpenAI account, twice per
window: once so the model can group the owner's messages into turns and write an assistant-style
draft for each usable turn, and once so a judge model can accept or reject every proposed pair.
Full contact names beyond first names, raw handles, chat identifiers, message identifiers, and
attachment contents are not sent; windows carry only an opaque window hash and small integer
indices. Entire windows of incoming messages are included, so review the provider's current data
policy and your account settings deliberately before running the command.

Both requests set `store=False`, which disables response storage for the request but does not
supersede the provider's account-level processing or retention terms. Responses are validated
mechanically (echoed identifiers, in-range owner-only message indices, one verdict per candidate);
windows whose responses stay malformed after retries are recorded as failed and excluded — there is
no local fallback that fabricates data. Results and review Markdown remain under ignored private
directories with restrictive permissions, and the aggregate report contains counts, hashes, and
token usage but no message text or API key.

## Censor stage

Generation never publishes training data. `censor-llm-dataset` is the only publisher of the
ready-to-train splits, and it screens every judge-accepted pair — draft, original messages, and
reviewer note together — through a hosted censor call (`store=False`) before publication. The
censor excludes, entirely rather than redacting:

- adult content: vulgar, violent, or sexual material; and
- secrets and super-personal information: passwords, API keys, secret keys, credentials,
  verification codes, financial numbers, government identifiers, and comparably sensitive data,
  on either side of the pair — a secret in the draft input is still a leak.

Publication is fail-closed: a pair whose screening never succeeds is excluded as
`unscreened_failure`, so nothing unscreened can reach `dataset/`. Exclusion counts by category
appear in the aggregate censor report, and every excluded row is rendered into a private
`censor-review.md` for spot-checking. That review file itself contains the sensitive text — it
exists so you can verify the drops — and stays under the ignored private directory. Screening
sends pair text to the provider under the same `store=False` terms as generation.

The commands run only when explicitly invoked with `OPENAI_API_KEY`. Adapter training, prediction,
and rewriting remain on the Mac. No command sends a generated message automatically.
