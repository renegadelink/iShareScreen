"""Frozen entry point. Keep CLI output and hide the console for the GUI."""
import multiprocessing
import sys


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Redirected Windows streams otherwise default to cp1252. The UI startup
    # message and media diagnostics contain Unicode, as can account passwords.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    if sys.platform == 'win32' and len(sys.argv) == 1:
        import ctypes
        console = ctypes.windll.kernel32.GetConsoleWindow()
        if console:
            ctypes.windll.user32.ShowWindow(console, 0)
    from isharescreen.tui.entry import main
    raise SystemExit(main())
