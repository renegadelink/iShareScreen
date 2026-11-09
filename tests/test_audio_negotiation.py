"""Audio capability diagnostics must distinguish real settings from placeholders."""
import os
import plistlib
import unittest
import zlib
from unittest.mock import patch

from isharescreen.proxy.protocol import offers
from isharescreen.proxy.protocol.audio_negotiation import audio_answer_fields


def audio_blob(values, *, video=False):
    stream = b"".join(offers._field_varint(k, v) for k, v in values.items())
    result = offers._field_bytes(3, stream)
    if video:
        result += offers._field_bytes(5, offers._field_varint(4, 1920))
    return result


class AudioNegotiationTests(unittest.TestCase):
    def test_extracts_numeric_audio_capabilities(self):
        fields = {1: 123, 2: 0, 3: 0, 4: 0x1600, 5: 0x600, 6: 1}
        self.assertEqual(audio_answer_fields(audio_blob(fields)), fields)

    def test_ignores_video_answer_audio_placeholder(self):
        self.assertIsNone(audio_answer_fields(audio_blob({1: 0, 4: 0}, video=True)))

    def test_ignores_missing_and_malformed_settings(self):
        for blob in (b"", b"\x1a\x80", b"\x1a\x05\x08", b"\x1a\x02\x20\x80"):
            with self.subTest(blob=blob):
                self.assertIsNone(audio_answer_fields(blob))

    @patch.dict(os.environ, {"ISS_AUDIO_F4": "0x5e7f"})
    def test_hex_override_preserves_default_offer(self):
        actual = offers._build_mediablob(8, 123, 456)
        with patch.dict(os.environ, {"ISS_AUDIO_F4": "24191"}):
            self.assertEqual(actual, offers._build_mediablob(8, 123, 456))

    def test_rejects_invalid_unsigned_mask(self):
        for value in ("-1", "4294967296", "banana"):
            with self.subTest(value=value), patch.dict(os.environ, {"ISS_AUDIO_F4": value}):
                with self.assertRaisesRegex(ValueError, "ISS_AUDIO_F4"):
                    offers._build_mediablob(8, 1, 1)

    def test_logs_audio_answer_from_embedded_plist_of_either_length_parity(self):
        blob = audio_blob({1: 123, 4: 0x1600, 5: 0x600, 6: 1})
        for padding in ("a", "ab"):
            plist = plistlib.dumps({"avcMediaStreamNegotiatorMediaBlob": zlib.compress(blob),
                                   "padding": padding}, fmt=plistlib.FMT_BINARY)
            with self.subTest(length=len(plist)), self.assertLogs(offers.log, level="INFO") as logs:
                self.assertEqual(offers.extract_canvas_dims(b"\x00" + plist), (0, 0, 0))
            self.assertIn("F3.f4=0x1600 F3.f5=0x600", " ".join(logs.output))
            self.assertNotIn("F3.f1=", " ".join(logs.output))


if __name__ == "__main__":
    unittest.main()
