/*
 * liquid-glass.js — the shell's behaviour. No dependencies; load with `defer`.
 *
 *  - Sidebar collapse, remembered per browser (#sb-collapse toggles
 *    body.sb-collapsed; aria-expanded and the label follow).
 *  - Server status dot in the sidebar foot (#connection-dot / #status-badge),
 *    from a cheap health endpoint — set data-health-url on <body> to enable
 *    (e.g. data-health-url="/healthz"). Pages with a live socket may update
 *    the same two elements; both report the same thing.
 *  - A 401 from any same-origin /api/ call sends the person to sign in
 *    (data-login-url on <body>, default /login) instead of leaving panels
 *    silently empty when a session expires mid-page.
 *
 * Every storage access is wrapped: private windows and blocked storage throw.
 */
(function () {
  const body = document.body;

  // ── sidebar collapse ────────────────────────────────────────────────────
  const KEY = 'liquid-glass.sidebar.collapsed';
  const btn = document.getElementById('sb-collapse');
  const setCollapsed = (on) => {
    body.classList.toggle('sb-collapsed', on);
    if (btn) {
      btn.setAttribute('aria-expanded', on ? 'false' : 'true');
      btn.setAttribute('aria-label', on ? 'Expand sidebar' : 'Collapse sidebar');
    }
  };
  let stored = null;
  try { stored = localStorage.getItem(KEY); } catch (e) { /* storage blocked */ }
  setCollapsed(stored === '1');
  if (btn) {
    btn.addEventListener('click', () => {
      const now = !body.classList.contains('sb-collapsed');
      setCollapsed(now);
      try { localStorage.setItem(KEY, now ? '1' : '0'); } catch (e) { /* ignore */ }
    });
  }

  // ── server status ───────────────────────────────────────────────────────
  const healthUrl = body.dataset.healthUrl;
  const dot = document.getElementById('connection-dot');
  const label = document.getElementById('status-badge');
  if (healthUrl && (dot || label)) {
    const check = async () => {
      let up = false;
      try { up = (await fetch(healthUrl, { cache: 'no-store' })).ok; } catch (e) { /* offline */ }
      if (dot) dot.className = up ? 'dot-on' : 'dot-off';
      if (label && (!up || label.textContent === 'Offline')) label.textContent = up ? 'Online' : 'Offline';
    };
    check();
    setInterval(check, 15000);
  }

  // ── expired session → sign in ───────────────────────────────────────────
  const loginUrl = body.dataset.loginUrl || '/login';
  const origFetch = window.fetch;
  window.fetch = async function (...args) {
    const res = await origFetch.apply(this, args);
    if (res.status === 401) {
      const url = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url) || '';
      if (url.startsWith('/api/') || url.startsWith(location.origin + '/api/')) {
        location.href = loginUrl + '?next=' + encodeURIComponent(location.pathname + location.search);
      }
    }
    return res;
  };
})();
