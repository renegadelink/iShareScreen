"""Browser integration check of the real local server, without a remote Mac.

Run: .venv/Scripts/python.exe packaging/check_ui.py
Requires the optional development tools: pip install playwright;
python -m playwright install chromium.
"""
from pathlib import Path
import http.server
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from playwright.sync_api import sync_playwright, expect
from isharescreen.gui import connect as C

out = Path('build/previews')
out.mkdir(parents=True, exist_ok=True)
C.load_last = lambda: None
C._PROBE_DONE.set()
server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), C._Handler)
C._GUI_PORT = server.server_port
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f'http://127.0.0.1:{server.server_port}'
errors = []
try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={'width':1440,'height':1080}, device_scale_factor=1)
        page.set_default_timeout(8000)
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(url)
        expect(page.get_by_role('heading', name='Make yourself at home.')).to_be_visible()
        assert not page.locator('form').evaluate('(f)=>f.checkValidity()')
        page.get_by_label('Mac address').fill('studio-mac.local')
        page.get_by_label('Username', exact=True).fill('studio')
        page.get_by_label('Password', exact=True).fill('test-only')
        assert page.locator('form').evaluate('(f)=>f.checkValidity()')
        page.get_by_role('button', name='Show password').click()
        expect(page.locator('#password')).to_have_attribute('type', 'text')
        page.get_by_role('button', name='Hide password').click()
        page.screenshot(path=str(out/'connect-desktop.png'),full_page=True)
        page.locator('summary').click()
        expect(page.get_by_label('Video decoder')).to_be_visible()
        page.get_by_role('button',name='Reduce effects').click()
        expect(page.locator('html')).to_have_attribute('data-effects','reduced')
        page.get_by_role('button',name='Restore effects').click()
        for width in (390,320):
            page.set_viewport_size({'width':width,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Overflow at {width}'
            expect(page.get_by_role('button',name='Connect to Mac')).to_be_visible()
            page.screenshot(path=str(out/f'connect-{width}.png'), full_page=True)
        page.emulate_media(reduced_motion='reduce')
        page.reload()
        expect(page.locator('html')).to_have_attribute('data-motion','reduced')
        page.goto(url+'/dashboard')
        expect(page.get_by_role('heading',name='Your Mac, in view.')).to_be_visible()
        # Exercise actual SSE delivery with clearly synthetic test data.
        C._emit('status', {'status':'running','host':'test-fixture.local','frontend':'browser'})
        C._emit('snapshot', {'decoder':'WebCodecs (test fixture)', 'uptime_s':12,
                'rx':{'video_pps':123,'video_mbps':8},'tiles':[{'fps':60}]})
        expect(page.locator('#s_dec')).to_have_text('WebCodecs (test fixture)')
        expect(page.locator('#n_vid')).to_contain_text('123 pps')
        C._emit('session_choice', {'console_user':'Test fixture'})
        expect(page.get_by_role('dialog')).to_be_visible()
        expect(page.get_by_role('button',name='Share their session')).to_be_focused()
        page.keyboard.press('Shift+Tab')
        expect(page.get_by_role('button',name='Sign in separately')).to_be_focused()
        page.get_by_role('button',name='Share their session').click()
        expect(page.get_by_role('dialog')).not_to_be_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(out/'dashboard-mobile.png'),full_page=True)
        page.set_viewport_size({'width':1440,'height':1000})
        page.screenshot(path=str(out/'dashboard-desktop.png'),full_page=True)
        page.get_by_role('button',name='Disconnect',exact=True).click()
        expect(page.locator('#s_status')).to_have_text('ended')
        page.goto(url+'/video-wait')
        expect(page.get_by_role('heading',name='Bringing your Mac closer.')).to_be_visible()
        assert not errors, errors
        browser.close()
    for path in ('/static/../connect.py','/static/missing.css'):
        try:
            urlopen(url+path)
            raise AssertionError('Unexpected static access: '+path)
        except HTTPError as error:
            assert error.code == 404
    try:
        urlopen(Request(url+'/quit',method='POST',headers={'Origin':'https://example.org'}))
        raise AssertionError('Cross-origin command was accepted')
    except HTTPError as error:
        assert error.code == 403
    print('UI checks passed: desktop, 390/320px, form, password, disclosure, effects, reduced motion, SSE, dialog focus, disconnect, waiting, static isolation, origin guard.')
finally:
    server.shutdown()
    server.server_close()
