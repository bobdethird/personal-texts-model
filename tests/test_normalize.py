from imessage_mlx.data.normalize import normalize_text


def test_normalizes_without_destroying_style() -> None:
    value = "  Héllo   WORLD 😊\r\ncall +1 (212) 555-0199 or a@example.com  "
    result = normalize_text(value, {"urls": True, "emails": True, "phone_numbers": True})
    assert result == "Héllo WORLD 😊\ncall <|phone|> or <|email|>"
