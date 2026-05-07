"""Native desktop frontend — glfw window + wgpu rendering.

Renders the iss decode output directly into a native window for the
lowest latency the decode pipeline can deliver.

Modules:

    app     `run(config, ...)` — CLI entry point; window + render loop
    gpu     `Renderer` + WGSL shader source — wgpu pipeline
    keymap  glfw → X11/RFB input translation tables
"""
