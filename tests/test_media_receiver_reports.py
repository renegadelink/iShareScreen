"""Audio and video liveness use authenticated reports on their own legs."""
import struct
import unittest
from unittest.mock import Mock, patch

from isharescreen.proxy.session import Session, SessionConfig
from isharescreen.proxy.protocol.srtp import SRTCPEncryptor, SRTCPDecryptor
import socket


class MediaReceiverReportTests(unittest.TestCase):
    def setUp(self):
        self.audio_key = bytes(range(46))
        self.video_key = bytes(reversed(range(46)))
        self.session = Session.__new__(Session)
        session = self.session
        session._sock_ctrl = Mock()
        session._sock_video = Mock()
        session._dest_host = "192.0.2.19"
        session._config = SessionConfig(host="192.0.2.19", username="unused", password="unused")
        session._our_audio_ssrc = 0x11223344
        session._our_video_ssrc = 0x55667788
        session._audio_srtcp_enc = SRTCPEncryptor.from_blob(self.audio_key)
        session._srtcp_enc = SRTCPEncryptor.from_blob(self.video_key)
        session._ssrc_to_tile = {}
        session._max_seq = {}
        session._roc = {}
        session._server_sr = {}
        session._tx_tick = 1

    def test_audio_keepalive_is_rtcp_with_audio_key_and_ssrc(self):
        self.session._send_heartbeat()
        packet, destination = self.session._sock_ctrl.sendto.call_args.args
        decoded = SRTCPDecryptor.from_blob(self.audio_key).unprotect(packet)
        self.assertEqual(decoded, struct.pack(">BBHI", 0x80, 201, 1, 0x11223344))
        self.assertEqual(destination, ("192.0.2.19", 5900))
        self.session._sock_video.sendto.assert_not_called()

    def test_video_report_uses_video_port_key_and_ssrc(self):
        self.session._send_rr_and_maybe_sr()
        self.session._sock_ctrl.sendto.assert_not_called()
        packet, destination = self.session._sock_video.sendto.call_args.args
        decoded = SRTCPDecryptor.from_blob(self.video_key).unprotect(packet)
        self.assertEqual(decoded, struct.pack(">BBHI", 0x80, 201, 1, 0x55667788))
        self.assertEqual(destination, ("192.0.2.19", 5901))

    def test_receiver_reports_continue_while_waiting_for_first_video(self):
        events = []
        clock = [0.0]

        def monotonic():
            clock[0] += 0.2
            return clock[0]

        def receive(_size):
            events.append("receive")
            raise socket.timeout()

        sock = Mock()
        sock.recvfrom.side_effect = receive
        with patch("isharescreen.proxy.session.time.monotonic", monotonic):
            Session._drain_socket_into(sock, [], max_seconds=2,
                                       on_tick=lambda: events.append("report"))
        self.assertEqual(events[0], "report")
        self.assertGreaterEqual(events.count("report"), 2)


if __name__ == "__main__":
    unittest.main()
