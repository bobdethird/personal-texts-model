import plistlib
from types import SimpleNamespace

from imessage_mlx.data.attributed_body import decode_attributed_body


def test_decodes_binary_plist_string() -> None:
    payload = plistlib.dumps({"metadata": "NSString", "value": "hello from archive 😊"})
    assert decode_attributed_body(payload) == "hello from archive 😊"


def test_decodes_utf8_segment_from_typed_stream_like_payload() -> None:
    payload = b"streamtyped\x00NSString\x00\x01this is the visible message\x00NSObject"
    assert decode_attributed_body(payload) == "this is the visible message"


def test_decodes_real_typedstream_nsstring_structure() -> None:
    payload = bytes.fromhex(
        "040b73747265616d747970656481e803840140848484084e53537472696e6701"
        "8484084e534f626a656374008584012b0c737472696e672076616c756586"
    )
    assert decode_attributed_body(payload) == "string value"


def test_prefers_attributed_string_backing_text_over_imessage_metadata(monkeypatch) -> None:
    archived = SimpleNamespace(
        contents=[
            SimpleNamespace(values=[SimpleNamespace(value="the actual visible reply")]),
            SimpleNamespace(values=[SimpleNamespace(value="__kIMMessagePartAttributeName")]),
        ]
    )
    monkeypatch.setattr(
        "imessage_mlx.data.attributed_body.typedstream.unarchive_from_data", lambda _: archived
    )
    payload = b"\x04\x0bstreamtyped" + b"synthetic"
    assert decode_attributed_body(payload) == "the actual visible reply"


def test_rejects_imessage_attribute_keys_as_visible_text(monkeypatch) -> None:
    archived = SimpleNamespace(
        contents=[
            SimpleNamespace(values=[SimpleNamespace(value="__kIMFileTransferGUIDAttributeName")])
        ]
    )
    monkeypatch.setattr(
        "imessage_mlx.data.attributed_body.typedstream.unarchive_from_data", lambda _: archived
    )
    payload = b"\x04\x0bstreamtyped" + b"synthetic"
    assert decode_attributed_body(payload) is None
