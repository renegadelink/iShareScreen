"""A new single-picture HEVC encoder must not inherit the previous DPB."""
import os
import unittest
from unittest.mock import patch

from test_ssrc_adoption import _session


class Hevc420GenerationTests(unittest.TestCase):
    @patch.dict(os.environ, {"ISS_TILES_PER_FRAME": "1", "ISS_HEVC_CHROMA": "420"})
    def test_new_420_ssrc_resets_references_inside_rapid_restart_guard(self):
        session = _session("hevc")
        session._note_unknown_ssrc(0x2000)
        self.assertEqual(session._ssrc_to_tile, {0x2000: 0})
        self.assertEqual(session._decoder.restart_calls, 1)
        self.assertEqual(session.fir_calls, 1)

    @patch.dict(os.environ, {"ISS_TILES_PER_FRAME": "1", "ISS_HEVC_CHROMA": "444"})
    def test_legacy_hevc_keeps_its_restart_guard(self):
        session = _session("hevc")
        session._note_unknown_ssrc(0x2000)
        self.assertEqual(session._decoder.restart_calls, 0)
        self.assertEqual(session.fir_calls, 1)

    @patch.dict(os.environ, {"ISS_TILES_PER_FRAME": "1", "ISS_HEVC_CHROMA": "420"})
    def test_resize_waits_for_new_parameter_sets_before_restarting(self):
        session = _session("hevc")
        session._needs_param_harvest = True
        session._note_unknown_ssrc(0x2000)
        self.assertEqual(session._decoder.restart_calls, 0)
        self.assertEqual(session.fir_calls, 1)


if __name__ == "__main__":
    unittest.main()
