"""iShareScreen — Apple Screen Sharing High Performance mode client library.

Layout (no symbols re-exported here — import directly from the modules):

    isharescreen.cli                        main()  — the `iss` console entry point
    isharescreen.proxy.session              Session, SessionConfig (the engine)
    isharescreen.proxy.input                InputController
    isharescreen.proxy.media.*              HEVC + AAC-ELD decoders, NAL utilities
    isharescreen.proxy.protocol.*           wire protocol (auth, SRTP, RFB, etc.)
    isharescreen.frontend.connect_prompt    launch-time terminal prompt
    isharescreen.frontend.desktop.*         native wgpu + glfw viewer
"""
__version__ = "0.1.0"
