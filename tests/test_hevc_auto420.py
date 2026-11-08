"""Automatic HEVC chroma selection must require real hardware support."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import av
import pytest

from isharescreen.proxy.media import hwcaps, registry


@pytest.mark.parametrize('available,expected', [
    ({'444', '420'}, ('hevc', '444')),
    ({'420'}, ('hevc', '420')),
    (set(), ('avc', '420')),
])
def test_auto_format_ladder(monkeypatch, available, expected):
    monkeypatch.delenv('ISS_HEVC_CHROMA', raising=False)
    monkeypatch.setattr(registry, 'can_decode', lambda codec, chroma, **kw: chroma in available)
    assert registry.resolve_video_format('auto') == expected
    assert 'ISS_HEVC_CHROMA' not in os.environ


def test_explicit_chroma_is_respected(monkeypatch):
    monkeypatch.setenv('ISS_HEVC_CHROMA', '444')
    monkeypatch.setattr(registry, 'can_decode', lambda codec, chroma, **kw: chroma == '420')
    assert registry.resolve_video_format('auto') == ('avc', '420')


def test_software_availability_is_not_hardware_support(monkeypatch):
    monkeypatch.setattr(hwcaps, 'supports_hevc420_hwdecode', lambda: False)
    assert registry.can_decode('hevc', '420')
    assert not registry.can_decode('hevc', '420', hardware_only=True)


def test_probe_sample_is_valid_plain_420():
    ctx = av.CodecContext.create('hevc', 'r')
    frames = ctx.decode(av.Packet(hwcaps._HEVC420_SAMPLE)) + ctx.decode(None)
    assert [(f.width, f.height, f.format.name) for f in frames] == [(64, 64, 'yuv420p')]


def test_hardware_probe_is_cached(monkeypatch):
    monkeypatch.setattr(hwcaps, '_cache', {})
    probe = Mock(return_value=True)
    monkeypatch.setattr(hwcaps, '_probe_one', probe)
    assert hwcaps.supports_hevc420_hwdecode()
    assert hwcaps.supports_hevc420_hwdecode()
    assert probe.call_count == 1
    assert probe.call_args.kwargs == {'chroma': '420'}


def test_probe_prohibits_software_fallback(monkeypatch):
    import av.codec.hwaccel
    hardware = Mock()
    monkeypatch.setattr(av.codec.hwaccel, 'HWAccel', hardware)
    ctx = Mock()
    ctx.decode.side_effect = [[SimpleNamespace(format=SimpleNamespace(name='nv12'))], []]
    create = Mock(return_value=ctx)
    monkeypatch.setattr(av, 'CodecContext', SimpleNamespace(create=create))
    assert hwcaps._probe_one('d3d11va', chroma='420')
    hardware.assert_called_once_with(device_type='d3d11va', allow_software_fallback=False)
    assert bytes(ctx.decode.call_args_list[0].args[0]) == hwcaps._HEVC420_SAMPLE


@patch.dict(os.environ)
def test_desktop_auto_applies_420_to_offer_and_decoder(monkeypatch):
    from isharescreen import cli
    monkeypatch.delenv('ISS_VIDEO_CODEC', raising=False)
    monkeypatch.delenv('ISS_HEVC_CHROMA', raising=False)
    monkeypatch.setattr(registry, 'resolve_video_format', lambda choice: ('hevc', '420'))
    run = Mock(return_value=0)
    monkeypatch.setitem(sys.modules, 'isharescreen.frontend.desktop.app', SimpleNamespace(run=run))
    args = SimpleNamespace(frontend='desktop', codec='auto', display=None, auto_quit_secs=None)
    assert cli._run_frontend(object(), args) == 0
    assert os.environ['ISS_VIDEO_CODEC'] == 'hevc'
    assert os.environ['ISS_HEVC_CHROMA'] == '420'
    assert registry.select('hevc').name == 'libav-hevc420'


@pytest.mark.parametrize('frontend', ['desktop', 'browser'])
@patch.dict(os.environ)
def test_gui_passes_auto_format_to_child_without_changing_parent(monkeypatch, frontend):
    from isharescreen.gui import connect
    monkeypatch.delenv('ISS_VIDEO_CODEC', raising=False)
    monkeypatch.delenv('ISS_HEVC_CHROMA', raising=False)
    resolve = Mock(return_value=('hevc', '420'))
    monkeypatch.setattr(registry, 'resolve_video_format', resolve)
    monkeypatch.setattr(connect, '_kill_current', Mock())
    monkeypatch.setattr(connect, '_free_bridge_port', Mock())
    monkeypatch.setattr(connect, '_emit', Mock())
    monkeypatch.setattr(connect, '_STATE', {})
    monkeypatch.setattr(connect, '_PROC', None)
    monkeypatch.setattr(connect, '_LAST', {})
    monkeypatch.setattr(connect, '_LAUNCH_N', 0)
    monkeypatch.setattr(connect.threading, 'Thread', Mock())
    popen = Mock()
    monkeypatch.setattr(connect.subprocess, 'Popen', popen)
    connect._launch({'host': 'example.invalid', 'user': 'test', 'password': 'fake',
                     'frontend': frontend, 'decoder': 'auto', 'audio': 'on'})
    cmd = popen.call_args.args[0]
    env = popen.call_args.kwargs['env']
    if frontend == 'desktop':
        assert cmd[cmd.index('--codec') + 1] == 'hevc'
        assert cmd[cmd.index('--decoder') + 1] == 'libav-hevc420'
        assert env['ISS_HEVC_CHROMA'] == '420'
        resolve.assert_called_once_with('auto')
    else:
        assert '--codec' not in cmd
        resolve.assert_not_called()
    assert 'ISS_HEVC_CHROMA' not in os.environ
