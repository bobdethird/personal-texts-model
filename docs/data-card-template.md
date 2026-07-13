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
- Accepted pair counts plus normalized/fuzzy, fact-conflict, and length removals
- Original and balanced exact/surface/high-overlap/substantive strata
- Local semantic model revision and content/style/fluency gate metrics
- Deterministic generation settings, local-only status, and rollback artifact
- Human comparison-review status and explicit no-automatic-sending warning
- Convergence schema, prompt, model, input, and generation fingerprints
- Stage A and Stage B token usage, retry counts, and no-text rejection aggregates
- Unique convergence target groups by split and complete rows by source register
- Local source-target semantic model revision, minimum/mean gates, and rejected group counts
- Same-challenge baseline/pilot fingerprint; per-register copy, fact, style, and convergence metrics
- Effective rows per target, target-exposure policy, and grouped human-review decision
- Explicit disclosure that complete target text was externally processed by Stage A with
  `store=False`, while Stage B received only sensitive semantic JSON

Do not include message samples, handles, chat identifiers, attachment names, or generated private
text in this document.
