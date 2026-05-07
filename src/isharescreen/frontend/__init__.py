"""User-facing frontends — what the human interacts with.

`connect_prompt` is the launch-time terminal prompt for host /
username / password. `desktop` is the native wgpu + glfw viewer
window that consumes decoded frames from the proxy and forwards
mouse + keyboard input back through it.

Layout:

    connect_prompt      `prompt(prefill) -> ConnectChoice`
    desktop.*           native wgpu viewer (`run(config, ...)`)
"""
