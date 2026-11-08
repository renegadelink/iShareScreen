"""An explicit HEVC request should be visible in the connect form's default."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from isharescreen.gui import connect


class HevcFrontendChoiceTests(unittest.TestCase):
    @patch.object(connect, "_decoder_options_html", return_value='')
    @patch.object(connect, "load_last", return_value=SimpleNamespace(
        host="example.invalid", user="example", frontend="browser"))
    @patch.dict(os.environ, {"ISS_VIDEO_CODEC": "hevc"})
    def test_hevc_request_defaults_to_desktop_over_saved_browser(self, *_):
        page = connect._form_page().decode()
        self.assertIn('value="desktop" selected', page)
        self.assertNotIn('value="browser" selected', page)

    @patch.object(connect, "_decoder_options_html", return_value='')
    @patch.object(connect, "load_last", return_value=SimpleNamespace(
        host="example.invalid", user="example", frontend="browser"))
    @patch.dict(os.environ, {"ISS_VIDEO_CODEC": "avc"})
    def test_h264_keeps_saved_browser_choice(self, *_):
        self.assertIn('value="browser" selected', connect._form_page().decode())


if __name__ == "__main__":
    unittest.main()
