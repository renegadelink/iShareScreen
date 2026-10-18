"""Windowed entry point for the Liquid Glass edition: the same app, opening in the glass look."""
import os
import sys
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    os.environ.setdefault('ISS_STYLE', 'glass')
    for name in ('stdin','stdout','stderr'):
        stream = getattr(sys,name)
        if stream is None:
            setattr(sys,name,open(os.devnull,'r' if name=='stdin' else 'w',encoding='utf-8'))
        elif hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='replace')
    from isharescreen.frontend.desktop.standard_app import main
    raise SystemExit(main())
