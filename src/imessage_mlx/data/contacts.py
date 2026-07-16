from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterable
from typing import Any


class ContactsAccessError(RuntimeError):
    """macOS Contacts could not be read for local sender-name enrichment."""


CONTACTS_JXA = r"""
function run() {
  const contacts = Application("Contacts");
  const records = [];
  for (const person of contacts.people()) {
    const name = person.name();
    const handles = [];
    for (const phone of person.phones()) handles.push(String(phone.value()));
    for (const email of person.emails()) handles.push(String(email.value()));
    if (name && handles.length) records.push({name: String(name), handles: handles});
  }
  return JSON.stringify(records);
}
""".strip()


def _handle_aliases(value: str) -> set[str]:
    handle = value.strip()
    if not handle:
        return set()
    if "@" in handle:
        return {f"email:{handle.casefold()}"}

    digits = re.sub(r"\D", "", handle)
    if len(digits) < 7:
        return {f"other:{handle.casefold()}"}

    aliases = {f"phone:{digits}"}
    if len(digits) == 10:
        aliases.add(f"phone:1{digits}")
    elif len(digits) == 11 and digits.startswith("1"):
        aliases.add(f"phone:{digits[1:]}")
    return aliases


def resolve_macos_contact_names(
    message_handles: Iterable[str],
    *,
    timeout_seconds: int = 120,
) -> dict[str, str]:
    """Map Messages handles to display names without persisting raw handles."""
    requested = {str(value) for value in message_handles if str(value).strip()}
    if not requested:
        return {}
    if shutil.which("osascript") is None:
        raise ContactsAccessError("Contact-name enrichment requires macOS and osascript")

    try:
        result = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", CONTACTS_JXA],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ContactsAccessError(
            f"Timed out reading macOS Contacts after {timeout_seconds} seconds"
        ) from error
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown Contacts error"
        raise ContactsAccessError(
            "Cannot read macOS Contacts. Grant Contacts access to the application running "
            f"this command, then retry. Detail: {detail}"
        )

    try:
        records: Any = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ContactsAccessError("macOS Contacts returned malformed data") from error
    if not isinstance(records, list):
        raise ContactsAccessError("macOS Contacts returned an unexpected result")

    names_by_alias: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        name = str(record.get("name", "")).strip()
        handles = record.get("handles", [])
        if not name or not isinstance(handles, list):
            continue
        for handle in handles:
            for alias in _handle_aliases(str(handle)):
                names_by_alias.setdefault(alias, name)

    resolved: dict[str, str] = {}
    for handle in requested:
        for alias in _handle_aliases(handle):
            if name := names_by_alias.get(alias):
                resolved[handle] = name
                break
    return resolved
