from imessage_mlx.data.redact import contains_obvious_pii, pseudonym, redact_text


def test_redacts_obvious_identifiers_and_uses_stable_namespaced_hashes() -> None:
    text = "a@example.com +1 212-555-0199 https://example.com/private"
    redacted = redact_text(text)
    assert redacted == "<|email|> <|phone|> <|url|>"
    assert not contains_obvious_pii(redacted)
    key = b"x" * 32
    assert pseudonym("alice", key, "handle") == pseudonym("alice", key, "handle")
    assert pseudonym("alice", key, "handle") != pseudonym("alice", key, "chat")
