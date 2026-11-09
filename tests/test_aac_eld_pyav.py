"""Plain-ELD playback and automatic backend selection."""
import os
import unittest
from unittest.mock import patch

import av
import numpy as np

from isharescreen.proxy.media import aac_eld


class PlainEldTests(unittest.TestCase):
    def test_real_decoder_opens_reference_configuration(self):
        decoder = aac_eld.make_aac_eld_decoder(prefer="pyav-eld")
        self.assertEqual(decoder._codec.extradata, bytes.fromhex("f8e65000"))
        self.assertEqual(decoder._codec.sample_rate, 48000)
        decoder.close()
        decoder.close()

    def test_decoded_pcm_matches_sink_shape_and_type(self):
        decoder = aac_eld.make_aac_eld_decoder(prefer="pyav-eld")
        pcm = np.full((2, 480), 0.25, dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(pcm, format="fltp", layout="stereo")
        frame.sample_rate = 48000
        with patch.object(decoder, "_codec") as codec:
            codec.decode.return_value = [frame]
            result = decoder.decode(b"example packet")
        self.assertEqual(result.shape, (480, 2))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, pcm.T)
        decoder.close()

    def test_bad_packet_does_not_crash_session(self):
        decoder = aac_eld.make_aac_eld_decoder(prefer="pyav-eld")
        self.assertIsNone(decoder.decode(b""))
        self.assertIsNone(decoder.decode(b"\xff"))
        decoder.close()

    @patch.dict(os.environ, {"AAC_BACKEND": ""})
    def test_windows_and_linux_default_to_pyav_without_loading_fdk(self):
        for platform in ("win32", "linux"):
            with self.subTest(platform=platform), \
                    patch.object(aac_eld.sys, "platform", platform), \
                    patch.object(aac_eld, "_PyAvEldBackend") as pyav, \
                    patch.object(aac_eld, "_LibFdkBackend") as fdk:
                self.assertIs(aac_eld.make_aac_eld_decoder(), pyav.return_value)
                pyav.assert_called_once_with()
                fdk.assert_not_called()

    @patch.dict(os.environ, {"AAC_BACKEND": ""})
    @patch.object(aac_eld.sys, "platform", "win32")
    @patch.object(aac_eld, "_PyAvEldBackend", side_effect=RuntimeError("unsupported"))
    @patch.object(aac_eld, "_LibFdkBackend")
    def test_falls_back_to_fdk_when_pyav_cannot_initialize(self, fdk, _pyav):
        self.assertIs(aac_eld.make_aac_eld_decoder(), fdk.return_value)
        _pyav.assert_called_once_with()
        fdk.assert_called_once_with()

    @patch.dict(os.environ, {"AAC_BACKEND": ""})
    @patch.object(aac_eld.sys, "platform", "win32")
    @patch.object(aac_eld, "_PyAvEldBackend", side_effect=RuntimeError("unsupported"))
    @patch.object(aac_eld, "_LibFdkBackend", side_effect=RuntimeError("missing"))
    def test_disables_audio_when_both_backends_fail(self, _fdk, _pyav):
        self.assertIsNone(aac_eld.make_aac_eld_decoder())
        _pyav.assert_called_once_with()
        _fdk.assert_called_once_with()

    @patch.dict(os.environ, {"AAC_BACKEND": ""})
    @patch.object(aac_eld.sys, "platform", "darwin")
    @patch.object(aac_eld, "_AudioToolboxBackend")
    @patch.object(aac_eld, "_PyAvEldBackend")
    def test_mac_still_prefers_audiotoolbox(self, pyav, toolbox):
        self.assertIs(aac_eld.make_aac_eld_decoder(), toolbox.return_value)
        pyav.assert_not_called()

    @patch.dict(os.environ, {"AAC_BACKEND": "fdk"})
    @patch.object(aac_eld, "_make_libfdk")
    @patch.object(aac_eld, "_PyAvEldBackend")
    def test_explicit_fdk_override_still_takes_precedence(self, pyav, fdk):
        self.assertIs(aac_eld.make_aac_eld_decoder(), fdk.return_value)
        pyav.assert_not_called()

    @patch.dict(os.environ, {"AAC_BACKEND": "fdk"})
    @patch.object(aac_eld, "_PyAvEldBackend", side_effect=RuntimeError("unsupported"))
    def test_explicit_argument_wins_and_reports_initialization_failure(self, _pyav):
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            aac_eld.make_aac_eld_decoder(prefer="pyav-eld")


if __name__ == "__main__":
    unittest.main()
