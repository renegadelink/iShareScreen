"""HEVC 4:2:0 negotiation must exclude unsupported chroma formats."""
import os
import unittest
import plistlib
import zlib
from unittest.mock import patch
from types import SimpleNamespace

from isharescreen.proxy.protocol import offers
from isharescreen.proxy.media import registry


def _fields(data):
    result = {}
    pos = 0
    while pos < len(data):
        tag, pos = offers._read_varint(data, pos)
        value, pos = offers._read_varint(data, pos)
        if tag & 7 == 2:
            value, pos = data[pos:pos + value], pos + value
        else:
            assert tag & 7 == 0
        result.setdefault(tag >> 3, []).append(value)
    return result


class Hevc420OfferTests(unittest.TestCase):
    @patch.dict(os.environ, {"ISS_VIDEO_CODEC": "hevc", "ISS_HEVC_CHROMA": "444"})
    def test_offer_excludes_444_without_changing_codec_bank(self):
        original = _fields(_fields(offers._build_mediablob(7, 1, 1))[5][0])
        os.environ["ISS_HEVC_CHROMA"] = "420"
        constrained = _fields(_fields(offers._build_mediablob(7, 1, 1))[5][0])
        self.assertEqual(original[8], [63])
        self.assertEqual(constrained[8], [3])
        self.assertEqual(_fields(constrained[3][0])[1], [100])  # still HEVC

    @patch.dict(os.environ, {"ISS_HEVC_CHROMA": "420"})
    @patch.object(registry.sys, "platform", "win32")
    @patch.object(registry, "_hevc444_method", return_value="qsv")
    def test_routes_to_generic_decoder_even_with_444_hardware(self, _probe):
        self.assertEqual(registry.select("hevc").name, "libav-hevc420")
        self.assertEqual(registry.normalize_override("libav", "hevc"), "libav-hevc420")
        self.assertEqual(registry.select("avc").name, "libav-avc420")

    @patch.dict(os.environ, {"ISS_HEVC_CHROMA": "444"})
    @patch.object(registry.sys, "platform", "win32")
    @patch.object(registry, "_hevc444_method", return_value=None)
    def test_default_hevc_does_not_select_420_decoder(self, _probe):
        self.assertEqual(registry.select("hevc").name, "libav-hevc444-sw")

    @patch.dict(os.environ, {"ISS_VIDEO_CODEC": "hevc", "ISS_HEVC_CHROMA": "420"})
    def test_420_defaults_to_one_picture_and_respects_explicit_tile_count(self):
        with patch.dict(os.environ):
            os.environ.pop("ISS_TILES_PER_FRAME", None)
            self.assertEqual(offers.tiles_per_frame(), 1)
            os.environ["ISS_TILES_PER_FRAME"] = "4"
            self.assertEqual(offers.tiles_per_frame(), 4)

    @patch.dict(os.environ, {"ISS_VIDEO_CODEC": "avc", "ISS_HEVC_CHROMA": "420"})
    def test_hevc_setting_does_not_change_avc_offer(self):
        stream = _fields(_fields(offers._build_mediablob(7, 1, 1))[5][0])
        self.assertEqual(stream[8], [63])

    @patch.dict(os.environ, {"ISS_VIDEO_CODEC": "hevc", "ISS_HEVC_CHROMA": "420",
                             "ISS_TILES_PER_FRAME": "1"})
    def test_single_picture_420_uses_complete_nearby_profile(self):
        video, audio = offers.create_offers()
        video = plistlib.loads(video)
        self.assertEqual(video["avcMediaStreamNegotiatorMode"], 9)
        self.assertEqual(video["avcMediaStreamNegotiatorDirection"], 2)
        self.assertEqual(plistlib.loads(audio)["avcMediaStreamNegotiatorMode"], 8)
        blob = _fields(zlib.decompress(video["avcMediaStreamNegotiatorMediaBlob"]))
        settings = _fields(blob[5][0])
        self.assertEqual(settings[2], [0])
        self.assertEqual(settings[6], [1])
        self.assertEqual(settings[8], [3])
        payload = _fields(settings[3][0])
        self.assertEqual(payload[1], [100])
        self.assertIn(b"SW:0;", payload[3][0])
        tiers = [_fields(value) for value in blob[9]]
        self.assertTrue(any(tier.get(3) == [4096] and tier[2] == [40_000_000]
                            for tier in tiers))

    @patch.dict(os.environ, {"ISS_VIDEO_CODEC": "hevc", "ISS_HEVC_CHROMA": "444"})
    def test_default_hevc_retains_remote_desktop_profile(self):
        video, _ = offers.create_offers()
        video = plistlib.loads(video)
        self.assertEqual(video["avcMediaStreamNegotiatorMode"], 7)
        self.assertNotIn("avcMediaStreamNegotiatorDirection", video)
        blob = _fields(zlib.decompress(video["avcMediaStreamNegotiatorMediaBlob"]))
        self.assertEqual(_fields(blob[5][0])[8], [63])

    @patch.dict(os.environ, {"ISS_HEVC_CHROMA": "420"})
    @patch.object(registry.sys, "platform", "win32")
    def test_rejects_444_override_for_420_stream(self):
        self.assertEqual(registry.select("hevc", override="qsv-hevc444").name,
                         "libav-hevc420")

    def test_downloaded_nv12_does_not_mislabel_strict_hardware_as_software(self):
        from isharescreen.proxy.media.hevc import HevcDecoder
        decoder = HevcDecoder.__new__(HevcDecoder)
        decoder._hw_verified = False
        decoder._hw_name = "d3d11va"
        decoder._hw_no_fallback = True
        decoder._pts_to_tile = {}
        decoder._pts_submit_t = {}
        frame = SimpleNamespace(format=SimpleNamespace(name="nv12"), pts=42)
        decoder._publish_frame(frame)
        self.assertEqual(decoder._hw_name, "d3d11va")


if __name__ == "__main__":
    unittest.main()
