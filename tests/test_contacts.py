import json
import subprocess

import pytest

from imessage_mlx.data import contacts
from imessage_mlx.data.contacts import ContactsAccessError, resolve_macos_contact_names


def test_resolves_phone_and_email_handles_without_returning_unmatched_contacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [
        {
            "name": "Alice Example",
            "handles": ["(212) 555-0199", "Alice@Example.com"],
        },
        {"name": "Unrequested Person", "handles": ["+1 415 555 0100"]},
    ]
    monkeypatch.setattr(contacts.shutil, "which", lambda _name: "/usr/bin/osascript")
    monkeypatch.setattr(
        contacts.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    resolved = resolve_macos_contact_names(
        ["+12125550199", "alice@example.com", "+13105550101"]
    )

    assert resolved == {
        "+12125550199": "Alice Example",
        "alice@example.com": "Alice Example",
    }


def test_reports_contacts_permission_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contacts.shutil, "which", lambda _name: "/usr/bin/osascript")
    monkeypatch.setattr(
        contacts.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Not authorized"
        ),
    )

    with pytest.raises(ContactsAccessError, match="Grant Contacts access"):
        resolve_macos_contact_names(["+12125550199"])
