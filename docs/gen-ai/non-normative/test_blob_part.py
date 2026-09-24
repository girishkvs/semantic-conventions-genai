"""BlobPart wire-format regressions.

Run from this directory with ``uv run python -m unittest -v test_blob_part``.
"""

import json
import unittest
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from models import (
    SCHEMAS,
    BlobPart,
    GenericPart,
    InputMessages,
    OutputMessages,
    TextPart,
)


class BlobPartTests(unittest.TestCase):
    samples = (
        (b"hello", "aGVsbG8="),
        (b"\x89PNG\r\n\x1a\n", "iVBORw0KGgo="),
        (b"", ""),
        (b"\xfb\xff", "+/8="),
        (b"aGVsbG8=", "YUdWc2JHOD0="),
    )

    def payload(self, content):
        return {
            "type": "blob",
            "modality": "image",
            "mime_type": "application/octet-stream",
            "content": content,
        }

    def test_python_entry_points_preserve_raw_bytes(self):
        adapter = TypeAdapter(BlobPart)
        for raw, _ in self.samples:
            with self.subTest(raw=raw):
                payload = self.payload(raw)
                parts = (
                    BlobPart(**payload),
                    BlobPart.model_validate(payload, strict=True),
                    adapter.validate_python(payload),
                )
                for part in parts:
                    self.assertEqual(part.content, raw)
                    self.assertEqual(part.model_dump()["content"], raw)
                    self.assertEqual(
                        BlobPart.model_validate(part.model_dump()).content, raw
                    )

    def test_python_strings_keep_their_utf8_behavior(self):
        for content in ("hello", "aGVsbG8=", "+/8=", "café"):
            with self.subTest(content=content):
                payload = self.payload(content)
                self.assertEqual(BlobPart(**payload).content, content.encode("utf-8"))
                self.assertEqual(
                    BlobPart.model_validate(payload).content, content.encode("utf-8")
                )

    def test_json_serialization_uses_standard_base64(self):
        adapter = TypeAdapter(BlobPart)
        for raw, encoded in self.samples:
            with self.subTest(raw=raw):
                part = BlobPart(**self.payload(raw))
                expected = self.payload(encoded)
                self.assertEqual(json.loads(part.model_dump_json()), expected)
                self.assertEqual(part.model_dump(mode="json"), expected)
                self.assertEqual(json.loads(adapter.dump_json(part)), expected)

    def test_json_entry_points_decode_standard_base64(self):
        adapter = TypeAdapter(BlobPart)
        for raw, encoded in self.samples:
            with self.subTest(raw=raw):
                wire = json.dumps(self.payload(encoded))
                self.assertEqual(BlobPart.model_validate_json(wire).content, raw)
                self.assertEqual(
                    BlobPart.model_validate_json(wire, strict=True).content, raw
                )
                self.assertEqual(adapter.validate_json(wire.encode()).content, raw)

    def test_json_round_trips_preserve_bytes(self):
        for raw, _ in self.samples:
            with self.subTest(raw=raw):
                part = BlobPart(**self.payload(raw))
                restored = BlobPart.model_validate_json(part.model_dump_json())
                self.assertEqual(restored, part)
                self.assertEqual(restored.content, raw)

    def test_input_and_output_message_round_trips(self):
        for model, role in ((InputMessages, "user"), (OutputMessages, "assistant")):
            for raw, encoded in self.samples:
                with self.subTest(model=model.__name__, raw=raw):
                    messages = model.model_validate(
                        [{"role": role, "parts": [self.payload(raw)]}]
                    )
                    self.assertIsInstance(messages.root[0].parts[0], BlobPart)
                    self.assertEqual(
                        json.loads(messages.model_dump_json())[0]["parts"][0],
                        self.payload(encoded),
                    )
                    wire = json.dumps(
                        [{"role": role, "parts": [self.payload(encoded)]}]
                    )
                    restored = model.model_validate_json(wire)
                    self.assertIsInstance(restored.root[0].parts[0], BlobPart)
                    self.assertEqual(restored.root[0].parts[0].content, raw)
                    self.assertEqual(restored, messages)

    def test_missing_null_and_empty_content_are_distinct(self):
        missing = self.payload(None)
        del missing["content"]
        for payload, error_type in (
            (missing, "missing"),
            (self.payload(None), "bytes_type"),
        ):
            for json_mode in (False, True):
                with self.subTest(payload=payload, json_mode=json_mode):
                    with self.assertRaises(ValidationError) as raised:
                        if json_mode:
                            BlobPart.model_validate_json(json.dumps(payload))
                        else:
                            BlobPart.model_validate(payload)
                    self.assertEqual(raised.exception.errors()[0]["loc"], ("content",))
                    self.assertEqual(raised.exception.errors()[0]["type"], error_type)

        part = BlobPart.model_validate_json(json.dumps(self.payload("")))
        self.assertEqual(part.content, b"")
        self.assertEqual(
            part.model_dump(
                mode="json",
                exclude_none=True,
                exclude_defaults=True,
                exclude_unset=True,
            )["content"],
            "",
        )

    def test_json_rejects_invalid_base64(self):
        for content in ("not base64!", "-_8=", "aGVsbG8", "aGVs\nbG8=", "é"):
            with self.subTest(content=content):
                with self.assertRaises(ValidationError) as raised:
                    BlobPart.model_validate_json(json.dumps(self.payload(content)))
                self.assertEqual(raised.exception.errors()[0]["loc"], ("content",))

    def test_non_blob_parts_keep_their_content(self):
        parts = [
            {"type": "text", "content": "+/8="},
            {"type": "vendor_note", "content": "+/8=", "custom": 1},
            {"type": "vendor_note", "content": ""},
            {"type": "vendor_note", "content": None},
            {"type": "vendor_note"},
        ]
        for model, role in ((InputMessages, "user"), (OutputMessages, "assistant")):
            with self.subTest(model=model.__name__):
                messages = model.model_validate_json(
                    json.dumps([{"role": role, "parts": parts}])
                )
                self.assertIsInstance(messages.root[0].parts[0], TextPart)
                for part in messages.root[0].parts[1:]:
                    self.assertIsInstance(part, GenericPart)
                self.assertEqual(
                    json.loads(messages.model_dump_json())[0]["parts"], parts
                )

    def test_blob_schema_describes_standard_base64_in_both_modes(self):
        for mode in ("validation", "serialization"):
            with self.subTest(mode=mode):
                schema = BlobPart.model_json_schema(mode=mode)
                schema.pop("$defs")
                content = schema["properties"]["content"]
                self.assertEqual(content["type"], "string")
                self.assertEqual(content.get("contentEncoding"), "base64")
                self.assertNotIn("format", content)
                self.assertNotIn("default", content)
                self.assertIn("content", schema["required"])
                for model in (InputMessages, OutputMessages):
                    self.assertEqual(
                        model.model_json_schema(mode=mode)["$defs"]["BlobPart"], schema
                    )

    def test_committed_schemas_match_generator(self):
        schema_dir = Path(__file__).resolve().parents[3] / "model" / "gen-ai"
        for filename, model in SCHEMAS.items():
            with self.subTest(filename=filename):
                expected = json.dumps(model.model_json_schema(), indent=4) + "\n"
                self.assertEqual(
                    (schema_dir / filename).read_text(encoding="utf-8"), expected
                )


if __name__ == "__main__":
    unittest.main()
