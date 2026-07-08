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

Do not include message samples, handles, chat identifiers, attachment names, or generated private
text in this document.
