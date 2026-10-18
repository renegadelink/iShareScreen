/* Pop/fade/slide transitions and responsive navigation; no per-frame JS loop.
 * Reduced motion follows the OS preference. An in-app "reduced" choice can
 * only strengthen that preference; it cannot turn OS reduced motion back on.
 */
(() => {
  const preference = matchMedia('(prefers-reduced-motion: reduce)');
  const MOTION_KEY = 'liquid-glass.pref.motion';
  const EFFECTS_KEY = 'liquid-glass.pref.effects';
  const active = new WeakMap();
  const running = new Set();
  const effects = {
    pop: { transform: 'translateY(8px) scale(.94)', opacity: 0 },
    fade: { transform: 'none', opacity: 0 },
    slide: { transform: 'translateY(16px)', opacity: 0 },
  };
  const readStored = (key) => {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  };
  const writeStored = (key, value) => {
    try { localStorage.setItem(key, value); } catch (e) { /* storage blocked */ }
  };
  function motionOff() {
    return preference.matches || document.documentElement.dataset.motion === 'reduced';
  }
  function applyMotionPreference() {
    const stored = readStored(MOTION_KEY) === 'reduced' ? 'reduced' : 'system';
    const reduced = preference.matches || stored === 'reduced';
    document.documentElement.dataset.motion = reduced ? 'reduced' : 'system';
    if (reduced) finishAll();
    return document.documentElement.dataset.motion;
  }
  function applyEffectsPreference() {
    const reduced = readStored(EFFECTS_KEY) === 'reduced';
    document.documentElement.dataset.effects = reduced ? 'reduced' : 'full';
    return document.documentElement.dataset.effects;
  }
  async function transition(el, visible, { effect = 'pop', duration, focusTarget } = {}) {
    if (!(el instanceof HTMLElement)) throw new TypeError('Motion target must be an HTML element.');
    if (!effects[effect]) throw new TypeError('Unknown motion effect.');
    if (duration !== undefined && (!Number.isFinite(duration) || duration < 0)) throw new TypeError('Invalid duration.');
    const reduced = motionOff();
    const effectName = reduced ? 'fade' : effect;
    const previous = active.get(el);
    const current = previous ? { opacity: getComputedStyle(el).opacity, transform: getComputedStyle(el).transform } : null;
    const originalWillChange = previous?.originalWillChange ?? el.style.willChange;
    previous?.animation.cancel();
    if (!visible && el.hidden && !previous) return { cancelled: false };
    const focusInside = !visible && el.contains(document.activeElement);
    if (visible) el.inert = false;
    else el.inert = true;
    el.hidden = false;
    el.dataset.lgMotion = effectName;
    const style = getComputedStyle(el);
    const ms = duration ?? (parseFloat(style.getPropertyValue('--motion-duration')) || 200);
    const playMs = reduced && ms > 0 ? Math.min(ms, 160) : ms;
    const chosen = effects[effectName];
    const from = current || (visible ? { ...chosen } : { opacity: style.opacity, transform: reduced ? 'none' : style.transform });
    const to = visible ? { opacity: 1, transform: 'none' } : { ...chosen };
    if (reduced) { from.transform = 'none'; to.transform = 'none'; }
    const finishVisibility = () => {
      if (!visible && focusInside) {
        if (focusTarget instanceof HTMLElement && focusTarget.isConnected) focusTarget.focus();
        else if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
      }
      el.hidden = !visible;
      if (visible) el.inert = false;
    };
    if (!el.animate || playMs === 0 || document.hidden) {
      active.delete(el);
      el.style.willChange = originalWillChange;
      finishVisibility();
      return { cancelled: false };
    }
    el.style.willChange = reduced ? 'opacity' : 'transform, opacity';
    const animation = el.animate([from, to], {
      duration: playMs,
      easing: visible ? 'cubic-bezier(.2,.8,.2,1)' : 'cubic-bezier(.4,0,1,1)',
      fill: 'both',
    });
    const state = { animation, originalWillChange };
    active.set(el, state); running.add(animation);
    try {
      await animation.finished;
      if (active.get(el) !== state) return { cancelled: true };
      finishVisibility();
      return { cancelled: false };
    } catch {
      return { cancelled: true };
    } finally {
      running.delete(animation);
      if (active.get(el) === state) {
        active.delete(el); el.style.willChange = originalWillChange; animation.cancel();
      }
    }
  }
  function finishAll() { for (const animation of running) animation.finish(); }
  function stagger(elements, visible, options = {}) {
    const list = Array.from(elements || []).filter((el) => el instanceof HTMLElement).slice(0, 6);
    const gap = Math.min(35, Math.max(20, Number.isFinite(options.gap) ? options.gap : 28));
    const reduced = motionOff();
    return Promise.all(list.map((el, index) => {
      const delay = reduced ? 0 : Math.min(150, index * gap);
      return new Promise((resolve) => setTimeout(resolve, delay)).then(() => transition(el, visible, options));
    }));
  }
  function feedback(el) {
    if (!(el instanceof HTMLElement) || !el.animate || document.hidden) return;
    el.animate([{ opacity: 0.55 }, { opacity: 1 }], {
      duration: motionOff() ? 120 : 180,
      easing: 'cubic-bezier(.2,.8,.2,1)',
    });
  }
  function reveal(elements, options = {}) {
    return stagger(elements, true, { effect: 'fade', ...options });
  }
  preference.addEventListener('change', applyMotionPreference);
  document.addEventListener('visibilitychange', () => { if (document.hidden) finishAll(); });
  document.addEventListener('click', (event) => {
    const link = event.target.closest?.('a[data-lg-transition]');
    if (!link || link.target || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) return;
    const url = new URL(link.href, location.href);
    if (url.origin !== location.origin) return;
    event.preventDefault();
    const reduced = motionOff();
    const animation = document.body.animate(
      reduced
        ? [{ opacity: 1 }, { opacity: 0 }]
        : [{ opacity: 1, transform: 'none' }, { opacity: 0, transform: 'translateY(8px)' }],
      { duration: reduced ? 140 : 200, easing: 'cubic-bezier(.4,0,1,1)', fill: 'forwards' }
    );
    animation.finished.finally(() => { location.href = link.href; });
  });
  applyMotionPreference();
  applyEffectsPreference();
  document.documentElement.classList.add('lg-motion');
  window.LiquidGlass = Object.assign(window.LiquidGlass || {}, {
    show: (el, options) => transition(el, true, options),
    hide: (el, options) => transition(el, false, options),
    stagger: (elements, options) => stagger(elements, true, options),
    reveal,
    feedback,
    setMotion(value) {
      writeStored(MOTION_KEY, value === 'reduced' ? 'reduced' : 'system');
      return applyMotionPreference();
    },
    setEffects(value) {
      writeStored(EFFECTS_KEY, value === 'reduced' ? 'reduced' : 'full');
      return applyEffectsPreference();
    },
    prefersReducedMotion: motionOff,
  });
  // Disclosure navigation: normal links, not an ARIA menu requiring arrow keys.
  for (const toggle of document.querySelectorAll('[data-glass-menu]')) {
    const panel = document.getElementById(toggle.dataset.glassMenu);
    if (!panel) continue;
    const mobile = matchMedia('(max-width: 760px)');
    let open = false;
    const setOpen = on => {
      open = on; toggle.setAttribute('aria-expanded', String(on));
      transition(panel, !mobile.matches || on, { effect: 'pop', focusTarget: toggle });
    };
    toggle.addEventListener('click', () => setOpen(!open));
    panel.addEventListener('click', event => { if (mobile.matches && event.target.closest('a')) setOpen(false); });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && mobile.matches && open) { setOpen(false); toggle.focus(); }
    });
    document.addEventListener('pointerdown', event => {
      if (mobile.matches && open && !panel.contains(event.target) && !toggle.contains(event.target)) setOpen(false);
    });
    const sync = () => { open = false; toggle.setAttribute('aria-expanded', 'false'); transition(panel, !mobile.matches, { duration: 0, focusTarget: toggle }); };
    mobile.addEventListener('change', sync); sync();
  }
})();
