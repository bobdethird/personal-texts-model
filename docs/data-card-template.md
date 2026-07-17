# Local iMessage model data card

## Scope

Private, local conversational text extracted from a read-only snapshot of the owner's Messages
database. Attachments are excluded.

## Required run-specific facts

- Snapshot and split hashes
- Retained and excluded row counts
- Body-recovery rate
- Train, validation, and test date ranges
- Token counts and model-selection calculation
- Model parameter count
- Validation and test perplexity
- Memorization-probe aggregates
- Adapter architecture, exact base-model revision, dependency pins, and adapter hash
- LLM dataset schema and prompt versions, proposal and judge model IDs
- Window counts: total, skipped, completed, failed (proposal/judge) with retry aggregates
- Example counts: proposed, accepted, rejected, and the judge acceptance rate
- Censor schema and prompt versions, censor model ID
- Censor counts: screened, allowed, excluded by category (adult content, sensitive secret,
  unscreened failure), and confirmation that published splits are censor-approved only
- Split sizes and the split fractions used
- Proposal, judge, and censor token usage
- Human review status of the generated pair sample, the censor exclusions, and held-out predictions
- Deterministic generation settings, local-only status, and no-automatic-sending warning
- Explicit disclosure that complete conversation windows, including incoming messages, were
  externally processed by the configured OpenAI models with `store=False`

Do not include message samples, handles, chat identifiers, attachment names, or generated private
text in this document.
