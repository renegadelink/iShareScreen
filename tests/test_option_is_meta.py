"""A Mac's Option key reaches the host as Meta, which screensharingd maps
to Option (it maps Alt to Command)."""
from __future__ import annotations

from pathlib import Path

import isharescreen
from isharescreen.frontend.desktop import keymap


def test_mac_option_keys_are_meta():
    assert keymap.option_keysyms("darwin") == (0xFFE7, 0xFFE8)


def test_pc_alt_keys_stay_alt():
    assert keymap.option_keysyms("win32") == (0xFFE9, 0xFFEA)
    assert keymap.option_keysyms("linux") == (0xFFE9, 0xFFEA)


def test_browser_sends_meta_for_mac_option():
    html = Path(isharescreen.__file__).parent / "frontend/wt/static/index.html"
    assert "? 0xffe7 : 0xffe9" in html.read_text()
