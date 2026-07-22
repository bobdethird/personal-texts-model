from __future__ import annotations

import re
import unicodedata

from imessage_mlx.data.redact import redact_text

HORIZONTAL_WHITESPACE_RE = re.compile(r"[\t\f\v ]+")
EXCESS_NEWLINES_RE = re.compile(r"\n{4,}")


def normalize_text(text: str, redaction: dict[str, bool] | None = None) -> str:
    value = unicodedata.normalize("NFC", text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\x00", "").replace("\ufffc", "")
    value = "\n".join(HORIZONTAL_WHITESPACE_RE.sub(" ", line).strip() for line in value.split("\n"))
    value = EXCESS_NEWLINES_RE.sub("\n\n\n", value).strip()
    if redaction is not None:
        value = redact_text(
            value,
            urls=bool(redaction.get("urls", True)),
            emails=bool(redaction.get("emails", True)),
            phone_numbers=bool(redaction.get("phone_numbers", True)),
        )
    return value
