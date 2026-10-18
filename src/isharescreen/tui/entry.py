"""`iss` console entry for the native standard-VNC desktop application."""
from __future__ import annotations

def main() -> int:
    # Normal launches use the native standard-VNC application. The old HP
    # protocol remains isolated in its module for development only.
    from isharescreen.frontend.desktop.standard_app import main as desktop_main
    return desktop_main()


if __name__ == "__main__":
    raise SystemExit(main())
