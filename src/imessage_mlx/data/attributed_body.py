from __future__ import annotations

import plistlib
import re
from collections.abc import Iterable
from typing import Any

import typedstream

METADATA_WORDS = {
    "NSMutableAttributedString",
    "NSAttributedString",
    "NSMutableString",
    "NSString",
    "NSDictionary",
    "NSObject",
    "NSFont",
    "NSColor",
    "streamtyped",
}


def _is_metadata_string(value: str) -> bool:
    stripped = value.strip()
    return (
        stripped in METADATA_WORDS
        or stripped.startswith("__kIM")
        or (stripped.startswith("kIM") and stripped.endswith("AttributeName"))
    )


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested in value.items():
            yield from _walk_strings(key)
            yield from _walk_strings(nested)


def _walk_decoded_object(
    value: Any, *, seen: set[int] | None = None, depth: int = 0
) -> Iterable[str]:
    """Collect strings from pytypedstream values without executing archived code."""
    if depth > 24:
        return
    if isinstance(value, str):
        yield value
        return
    if value is None or isinstance(value, (bytes, bytearray, memoryview, int, float, bool)):
        return
    seen = seen or set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from _walk_decoded_object(key, seen=seen, depth=depth + 1)
            yield from _walk_decoded_object(nested, seen=seen, depth=depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _walk_decoded_object(nested, seen=seen, depth=depth + 1)
    elif hasattr(value, "__dict__"):
        for name, nested in vars(value).items():
            if name in {"clazz", "class_name", "type_encoding"}:
                continue
            yield from _walk_decoded_object(nested, seen=seen, depth=depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _walk_strings(nested)


def _score(candidate: str) -> tuple[int, int, int]:
    stripped = candidate.strip()
    natural = sum(character.islower() or character.isspace() for character in stripped)
    return (natural, len(stripped), -sum(character == "_" for character in stripped))


def _best_candidate(candidates: Iterable[str]) -> str | None:
    cleaned = []
    for candidate in candidates:
        value = candidate.strip("\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\r ")
        if len(value) < 1 or _is_metadata_string(value):
            continue
        if all(not character.isprintable() and not character.isspace() for character in value):
            continue
        cleaned.append(value)
    return max(cleaned, key=_score, default=None)


def decode_attributed_body(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return None
    payload = bytes(value)
    if not payload:
        return None

    try:
        plist = plistlib.loads(payload)
    except (plistlib.InvalidFileException, ValueError, TypeError, OverflowError):
        plist = None
    if plist is not None:
        candidate = _best_candidate(_walk_strings(plist))
        if candidate:
            return candidate

    if payload.startswith(b"\x04\x0bstreamtyped"):
        try:
            archived = typedstream.unarchive_from_data(payload)
        except Exception:  # Undocumented archives can contain unsupported classes or encodings.
            archived = None
        if archived is not None:
            # NSMutableAttributedString stores its visible backing NSString in the first
            # archived field. Attribute dictionaries follow and contain internal keys such as
            # __kIMMessagePartAttributeName, which must never be treated as message text.
            contents = getattr(archived, "contents", None)
            if contents:
                candidate = _best_candidate(_walk_decoded_object(contents[0]))
                if candidate:
                    return candidate
            candidate = _best_candidate(_walk_decoded_object(archived))
            if candidate:
                return candidate
            return None

    decoded = payload.decode("utf-8", errors="ignore")
    chunks = re.split(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", decoded)
    return _best_candidate(chunks)
