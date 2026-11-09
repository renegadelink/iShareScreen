"""Transport announcements must remain visible before the RTP burst starts."""
import struct
import unittest
import socket
from unittest.mock import Mock

from isharescreen.proxy.protocol.media_diagnostics import parse_media_reply


class MediaStartupDiagnosticsTests(unittest.TestCase):
    def test_ports_include_enabled_flags_and_each_leg(self):
        body = struct.pack(">HHIHIHIHI", 1, 1, 0, 5900, 1, 5911, 3, 5912, 0)
        reply = parse_media_reply(struct.pack(">H", len(body)) + body)
        self.assertEqual(reply, {
            "kind": "ports", "audio_port": 5900, "audio_flags": 1,
            "video_port": 5911, "video_flags": 3,
            "video2_port": 5912, "video2_flags": 0,
        })

    def test_server_refusal_exposes_error_codes(self):
        body = struct.pack(">HHIII", 3, 2, 0, 2, 9)
        reply = parse_media_reply(struct.pack(">H", len(body)) + body)
        self.assertEqual(reply, {"kind": "error", "type": 2, "subcode": 9})

    def test_truncated_and_non_transport_replies_are_ignored(self):
        self.assertIsNone(parse_media_reply(b"\x00"))
        self.assertIsNone(parse_media_reply(b"\x00\x24\x00\x01"))
        body = struct.pack(">HHI", 2, 2, 0)
        self.assertIsNone(parse_media_reply(struct.pack(">H", len(body)) + body))

    def test_handshake_drain_logs_ports_before_discarding_messages(self):
        from isharescreen.proxy.protocol.negotiation import _drain_through_cipher
        body = struct.pack(">HHIHIHIHI", 1, 1, 0, 5900, 1, 5911, 1, 0, 0)
        rect = struct.pack(">HHHHiH", 0, 0, 0, 0, 1010, len(body)) + body
        message = b"\x00\x00\x00\x01" + rect
        sock = Mock()
        sock.recv.side_effect = [b"encrypted", socket.timeout()]
        cipher = Mock()
        cipher.decrypt_stream.return_value = ([message], b"")
        with self.assertLogs("isharescreen.proxy.protocol.media_diagnostics", "INFO") as logs:
            _drain_through_cipher(sock, cipher, timeout=0.5)
        self.assertIn("video=5911 flags=0x1", logs.output[0])
        cipher.decrypt_stream.assert_called_once_with(b"encrypted")

    def test_unknown_rectangle_does_not_log_embedded_false_announcement(self):
        from isharescreen.proxy.protocol.media_diagnostics import log_media_transport
        body = struct.pack(">HHIHIHIHI", 1, 1, 0, 5900, 1, 5911, 1, 0, 0)
        rect = struct.pack(">HHHHiH", 0, 0, 0, 0, 9999, len(body)) + body
        with self.assertNoLogs("isharescreen.proxy.protocol.media_diagnostics"):
            log_media_transport(b"\x00\x00\x00\x01" + rect)


if __name__ == "__main__":
    unittest.main()
