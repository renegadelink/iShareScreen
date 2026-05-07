"""WGPU planar-YUV renderer.

Three single-channel textures (Y, U, V) at full canvas resolution, one
fragment shader doing BT.709 *full-range* YUV→RGB, one full-screen
triangle. That's all.

Each tile's planes get written into a horizontal slice of the textures
via `upload_tile`; we don't allocate or compose a CPU-side full-canvas
buffer. Apple's HEVC RExt 4:4:4 source stays 4:4:4 end-to-end.
"""
from __future__ import annotations

import logging

import numpy as np
import wgpu


log = logging.getLogger(__name__)


WGSL = """
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    // One triangle that covers the viewport. UV maps so the visible
    // area [0,1]x[0,1] aligns with the texture.
    let xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(xy[i], 0.0, 1.0);
    o.uv = uv[i];
    return o;
}

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var u_tex: texture_2d<f32>;
@group(0) @binding(2) var v_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let y  = textureSample(y_tex, samp, in.uv).r;
    let cb = textureSample(u_tex, samp, in.uv).r - 0.5;
    let cr = textureSample(v_tex, samp, in.uv).r - 0.5;
    let r = y + 1.5748   * cr;
    let g = y - 0.187324 * cb - 0.468124 * cr;
    let b = y + 1.8556   * cb;
    return vec4<f32>(r, g, b, 1.0);
}
"""


class Renderer:
    """Build once, call `upload_tile` per fresh tile, then `draw`."""

    def __init__(
        self, device: wgpu.GPUDevice, surface_format: wgpu.TextureFormat,
        canvas_w: int, canvas_h: int,
    ) -> None:
        self._device = device
        self._w = canvas_w
        self._h = canvas_h

        tex_kw = dict(
            format=wgpu.TextureFormat.r8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            size=(canvas_w, canvas_h, 1),
        )
        self._y_tex = device.create_texture(**tex_kw)
        self._u_tex = device.create_texture(**tex_kw)
        self._v_tex = device.create_texture(**tex_kw)
        # WGPU does not guarantee initialised texture contents. On Mesa
        # i915 we observed unwritten Y/U/V regions sample as bright
        # green. Pre-fill: Y=0 + UV=128 → BT.709 full-range black.
        zeros_y = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        neutral_uv = np.full((canvas_h, canvas_w), 128, dtype=np.uint8)
        device.queue.write_texture(
            {"texture": self._y_tex, "origin": (0, 0, 0)}, zeros_y,
            {"offset": 0, "bytes_per_row": canvas_w},
            (canvas_w, canvas_h, 1),
        )
        for tex in (self._u_tex, self._v_tex):
            device.queue.write_texture(
                {"texture": tex, "origin": (0, 0, 0)}, neutral_uv,
                {"offset": 0, "bytes_per_row": canvas_w},
                (canvas_w, canvas_h, 1),
            )
        log.info(
            "renderer: 3x r8unorm %dx%d (%.1f MB total) — pre-cleared to black",
            canvas_w, canvas_h, 3 * canvas_w * canvas_h / 1e6,
        )

        sampler = device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
        )
        shader = device.create_shader_module(code=WGSL)
        tex_entry = {
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "texture": {"sample_type": wgpu.TextureSampleType.float, "view_dimension": "2d"},
        }
        bind_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, **tex_entry},
            {"binding": 1, **tex_entry},
            {"binding": 2, **tex_entry},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {}},
        ])
        self._pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[bind_layout]),
            vertex={"module": shader, "entry_point": "vs"},
            fragment={
                "module": shader, "entry_point": "fs",
                "targets": [{"format": surface_format}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )
        self._bind_group = device.create_bind_group(
            layout=bind_layout,
            entries=[
                {"binding": 0, "resource": self._y_tex.create_view()},
                {"binding": 1, "resource": self._u_tex.create_view()},
                {"binding": 2, "resource": self._v_tex.create_view()},
                {"binding": 3, "resource": sampler},
            ],
        )

    def upload_tile(self, tile_index: int, tile, slot_height: int) -> None:
        """Write tile's Y/U/V into the canvas-tile slice at row
        `tile_index * slot_height`.

        `slot_height` should be the encoder's actual CTU-padded picture
        height (typically `tile.height`, e.g. 304 rows for a 1920×1200/
        4-tile canvas). The first 3 tiles fill `slot_height` rows each;
        the last tile is bounded by `canvas_h - tile_index * slot_height`
        so we don't overrun the texture and so the bottom CTU-padding
        rows of tile 3 (Apple Screen Sharing also doesn't render those — they're
        encoder padding past `canvas_h`) are dropped from the display.
        """
        w = tile.width
        origin_y = tile_index * slot_height
        remaining = max(0, self._h - origin_y)
        rows = min(tile.height, slot_height, remaining)
        if rows <= 0:
            return
        origin = (0, origin_y, 0)

        y = np.frombuffer(tile.y, dtype=np.uint8)
        if tile.y_stride == w:
            y = y[: w * tile.height].reshape(tile.height, w)
        else:
            y = y[: tile.y_stride * tile.height].reshape(
                tile.height, tile.y_stride)[:, :w]
        self._device.queue.write_texture(
            {"texture": self._y_tex, "origin": origin},
            np.ascontiguousarray(y[:rows]),
            {"offset": 0, "bytes_per_row": w},
            (w, rows, 1),
        )

        if tile.v is None:  # NV-style biplanar — not produced by our decoder path
            return
        cw, ch = tile.chroma_width, tile.chroma_height
        # Same canvas-bound for chroma as for luma so the last tile's
        # padding chroma rows don't get uploaded.
        chroma_rows = min(ch, slot_height, max(0, self._h - origin_y))
        if chroma_rows <= 0:
            return
        u = np.frombuffer(tile.u, dtype=np.uint8)
        v = np.frombuffer(tile.v, dtype=np.uint8)
        if tile.uv_stride == cw:
            u = u[: cw * ch].reshape(ch, cw)
            v = v[: cw * ch].reshape(ch, cw)
        else:
            u = u[: tile.uv_stride * ch].reshape(ch, tile.uv_stride)[:, :cw]
            v = v[: tile.uv_stride * ch].reshape(ch, tile.uv_stride)[:, :cw]
        self._device.queue.write_texture(
            {"texture": self._u_tex, "origin": origin},
            np.ascontiguousarray(u[:chroma_rows]),
            {"offset": 0, "bytes_per_row": cw},
            (cw, chroma_rows, 1),
        )
        self._device.queue.write_texture(
            {"texture": self._v_tex, "origin": origin},
            np.ascontiguousarray(v[:chroma_rows]),
            {"offset": 0, "bytes_per_row": cw},
            (cw, chroma_rows, 1),
        )

    def draw(
        self, target_view: wgpu.GPUTextureView,
        viewport: tuple[float, float, float, float],
    ) -> None:
        """Render one frame into `target_view` clipped to `viewport`
        (`(x, y, w, h)` in surface pixels). The area outside `viewport`
        is cleared to black, which is the letterbox."""
        encoder = self._device.create_command_encoder()
        rpass = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0, 0, 0, 1),
        }])
        rpass.set_pipeline(self._pipeline)
        rpass.set_bind_group(0, self._bind_group)
        rpass.set_viewport(*viewport, 0.0, 1.0)
        rpass.draw(3)
        rpass.end()
        self._device.queue.submit([encoder.finish()])


def letterbox(
    canvas_w: int, canvas_h: int, surf_w: int, surf_h: int,
) -> tuple[float, float, float, float]:
    """Largest aspect-preserving viewport for `(canvas_w, canvas_h)`
    inside `(surf_w, surf_h)`. Returns `(x, y, w, h)` in pixels."""
    src = canvas_w / canvas_h
    dst = surf_w / surf_h
    if dst > src:
        h = float(surf_h)
        w = h * src
        return ((surf_w - w) / 2.0, 0.0, w, h)
    w = float(surf_w)
    h = w / src
    return (0.0, (surf_h - h) / 2.0, w, h)


__all__ = ["Renderer", "letterbox", "WGSL"]
