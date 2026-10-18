/* Liquid Glass button feedback. Load with defer; no dependencies.
 * Opt in with .btn-glass. Native button/link activation stays with the browser.
 * Dragging is visual feedback, not a draggable control or an action gesture.
 */
(() => {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const motionReduced = () => reduced.matches || document.documentElement.dataset.motion === 'reduced';
  const blocked = new WeakSet();
  let active = null;
  let frame = 0;
  const disabled = el => el.matches(':disabled, [aria-disabled="true"]');
  const clamp = (value, limit) => Math.max(-limit, Math.min(limit, value));
  const paint = () => {
    frame = 0;
    if (!active) return;
    const { el, rect, x, y } = active;
    if (!el.isConnected || disabled(el)) { reset(); return; }
    if (motionReduced()) return;
    // Highlight only. The button box stays put so the hit target does not move.
    el.style.setProperty('--glass-x', `${clamp(x - rect.left, rect.width)}px`);
    el.style.setProperty('--glass-y', `${clamp(y - rect.top, rect.height)}px`);
  };
  function reset() {
    cancelAnimationFrame(frame);
    frame = 0;
    if (!active) return;
    const el = active.el;
    el.classList.remove('glass-held');
    active = null;
  }
  document.addEventListener('pointerdown', event => {
    if (!event.isPrimary || event.button !== 0 || active) return;
    const el = event.target.closest?.('.btn-glass');
    if (!el) return;
    blocked.delete(el);
    if (disabled(el)) return;
    active = { el, id: event.pointerId, rect: el.getBoundingClientRect(),
      startX: event.clientX, startY: event.clientY,
      x: event.clientX, y: event.clientY };
    el.classList.add('glass-held');
    paint();
  });
  document.addEventListener('pointermove', event => {
    if (!active || event.pointerId !== active.id) return;
    active.x = event.clientX;
    active.y = event.clientY;
    if (Math.hypot(active.x - active.startX, active.y - active.startY) > 8) {
      blocked.add(active.el);
    }
    if (!frame) frame = requestAnimationFrame(paint);
  }, { passive: true });
  document.addEventListener('pointerup', event => {
    if (active?.id === event.pointerId) {
      if (Math.hypot(event.clientX - active.startX, event.clientY - active.startY) > 8) {
        blocked.add(active.el);
      }
      reset();
    }
  });
  document.addEventListener('pointercancel', event => {
    if (active?.id === event.pointerId) { blocked.add(active.el); reset(); }
  });
  document.addEventListener('click', event => {
    const el = event.target.closest?.('.btn-glass');
    if (el && (disabled(el) || (event.detail !== 0 && blocked.has(el)))) {
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }, true);
  window.addEventListener('blur', reset);
  window.addEventListener('pagehide', reset);
  window.addEventListener('resize', reset);
  document.addEventListener('scroll', reset, { passive: true, capture: true });
  document.addEventListener('visibilitychange', () => { if (document.hidden) reset(); });
  reduced.addEventListener('change', reset);
  new MutationObserver(reset).observe(document.documentElement, { attributes: true, attributeFilter: ['data-motion'] });
})();
