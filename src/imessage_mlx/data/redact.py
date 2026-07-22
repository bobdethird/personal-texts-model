from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from pathlib import Path

from imessage_mlx.utils import ensure_private_dir

URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)")


def load_or_create_key(path: str | Path) -> bytes:
    key_path = Path(path)
    ensure_private_dir(key_path.parent)
    if key_path.exists():
        key_path.chmod(0o600)
        key = key_path.read_bytes()
        if len(key) < 32:
            raise ValueError(f"Pseudonym key is unexpectedly short: {key_path}")
        return key
    key = secrets.token_bytes(32)
    descriptor = key_path.open("xb")
    try:
        descriptor.write(key)
        descriptor.flush()
    finally:
        descriptor.close()
    key_path.chmod(0o600)
    return key


def pseudonym(value: object, key: bytes, namespace: str) -> str:
    payload = f"{namespace}\0{value}".encode("utf-8", errors="surrogatepass")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()[:24]


def redact_text(
    text: str, *, urls: bool = True, emails: bool = True, phone_numbers: bool = True
) -> str:
    if urls:
        text = URL_RE.sub("<|url|>", text)
    if emails:
        text = EMAIL_RE.sub("<|email|>", text)
    if phone_numbers:
        text = PHONE_RE.sub("<|phone|>", text)
    return text


def contains_obvious_pii(text: str) -> bool:
    return bool(URL_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text))
