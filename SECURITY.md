# Security and privacy

This project operates on private communications. Treat every generated dataset, tokenizer,
checkpoint, evaluation prompt, and model as sensitive—even after pseudonymization.

## Never include private artifacts in a report

When opening an issue or security report, do not attach:

- `chat.db`, its WAL/SHM files, or any database snapshot
- Files from `work/` or `outputs/`
- Raw or processed message text
- Contact identifiers, pseudonym keys, prompts, or generated private replies
- Tokenizers, checkpoints, or final model weights trained on private messages

Use synthetic reproductions and aggregate counts only.

## Reporting a vulnerability

While the repository is private, report security concerns directly to the repository owner. Before
making the repository public, enable GitHub private vulnerability reporting and use that channel for
future reports.

If you accidentally commit private data, stop immediately. Do not merely add it to `.gitignore` in a
later commit—the data remains in Git history. Rotate any exposed secrets, remove the data from Git
history, and replace the remote repository before publication.
