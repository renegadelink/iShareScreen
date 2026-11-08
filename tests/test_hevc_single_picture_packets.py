"""Single-picture HEVC uses RTP framing without decoding-order fields."""
import os
import struct
import unittest
from unittest.mock import patch

from isharescreen.proxy.media.nalu import reassemble_group
from isharescreen.proxy.protocol.burst import gather_initial_burst


VPS = b"\x40\x01\xab\xcd"
SPS = b"\x42\x01\xef\x12"
PPS = b"\x44\x01\x80"
AP = b"\x60\x01" + b"".join(struct.pack(">H", len(n)) + n for n in (VPS, SPS, PPS))
FRAGMENTS = [b"\x62\x01\x94\xaa\xbb", b"\x62\x01\x54\xcc\xdd"]


class HevcSinglePicturePacketTests(unittest.TestCase):
    def test_aggregation_keeps_parameter_sets_without_donl(self):
        self.assertEqual(reassemble_group([AP], has_donl=False), [VPS, SPS, PPS])

    def test_fragmentation_keeps_first_two_bytes_of_each_fragment(self):
        self.assertEqual(reassemble_group(FRAGMENTS, has_donl=False),
                         [b"\x28\x01\xaa\xbb\xcc\xdd"])

    def test_legacy_multi_tile_aggregation_retains_donl_behavior(self):
        with_donl = AP[:2] + b"\x00\x09" + AP[2:]
        self.assertEqual(reassemble_group([with_donl]), [VPS, SPS, PPS])

    def test_legacy_fragmentation_retains_donl_behavior(self):
        fragments = [p[:3] + b"\x00\x09" + p[3:] for p in FRAGMENTS]
        self.assertEqual(reassemble_group(fragments), [b"\x28\x01\xaa\xbb\xcc\xdd"])

    @patch.dict(os.environ, {"ISS_TILES_PER_FRAME": "1"})
    def test_unmarked_parameter_packet_on_earlier_timestamp_is_harvested(self):
        payloads = [AP] + FRAGMENTS
        packets = [struct.pack(">BBHII", 0x80, 100 | (0x80 if i == 2 else 0),
                               i, 999 if i == 0 else 1000, 99) + data
                   for i, data in enumerate(payloads)]

        class DecryptedPackets:
            def decrypt(self, packet):
                return packet[:12], packet[12:]

            def get_primary_ssrc_group(self, tier=0):
                return {99}

        burst = gather_initial_burst(packets, DecryptedPackets(), min_packets=0,
                                     settle_seconds=0, deadline_seconds=0, codec="hevc")
        self.assertEqual((burst.vps, burst.sps, burst.all_pps[0]), (VPS, SPS, PPS))

    @patch.dict(os.environ, {"ISS_VIDEO_CODEC": "hevc", "ISS_HEVC_CHROMA": "420",
                             "ISS_TILES_PER_FRAME": "1"})
    def test_burst_harvests_single_picture_hevc_and_identifies_payload_100(self):
        payloads = [AP] + FRAGMENTS
        packets = [struct.pack(">BBHII", 0x80, 100 | (0x80 if i == 2 else 0),
                               i, 1000, 99) + data for i, data in enumerate(payloads)]

        class DecryptedPackets:
            def decrypt(self, packet):
                return packet[:12], packet[12:]

            def get_primary_ssrc_group(self, tier=0):
                return {99}

        with self.assertLogs("isharescreen.proxy.protocol.burst", "INFO") as logs:
            burst = gather_initial_burst(packets, DecryptedPackets(), min_packets=0,
                                         settle_seconds=0, deadline_seconds=0, codec="hevc")
        self.assertEqual((burst.vps, burst.sps, burst.all_pps[0]), (VPS, SPS, PPS))
        self.assertEqual(burst.last_idr[0], b"\x28\x01\xaa\xbb\xcc\xdd")
        self.assertTrue(any("-> HEVC" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
