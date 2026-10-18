/* Operational helpers for truthful input, readings, progress, and commands.
 * No dependencies. Domain limits below are demonstration contracts, not
 * equipment recommendations. Authorize and validate again on the server.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.LiquidGlass = Object.assign(root.LiquidGlass || {}, api);
})(typeof window !== 'undefined' ? window : null, function () {
  function commandReducer(state, event) {
    if (!state || !event) return state;
    if (event.type === 'submit') {
      if (state.phase !== 'idle' && state.phase !== 'verified' && state.phase !== 'failed') return state;
      return { phase: 'submitting', requestId: event.requestId, assetId: event.assetId };
    }
    if (state.phase === 'idle' || state.requestId !== event.requestId) return state;
    if (state.phase === 'verified' || state.phase === 'failed') return state;
    if (event.type === 'accepted') {
      return state.phase === 'submitting' ? { ...state, phase: 'accepted' } : state;
    }
    if (event.type === 'timeout') return { ...state, phase: 'unknown' };
    if (event.type === 'verified') {
      return {
        phase: 'verified', requestId: state.requestId, assetId: state.assetId,
        verifiedAt: event.verifiedAt,
      };
    }
    if (event.type === 'failed') {
      return { phase: 'failed', requestId: state.requestId, assetId: state.assetId, reason: event.reason };
    }
    return state;
  }

  function commandLabel(state) {
    switch (state && state.phase) {
      case 'submitting': return 'Requesting';
      case 'accepted': return 'Request accepted. Verification pending.';
      case 'unknown': return `Unknown outcome. Reconcile request ${state.requestId} before any retry.`;
      case 'verified': return 'Verified by the server.';
      case 'failed': return state.reason || 'Failed';
      default: return 'Idle';
    }
  }

  // Decimal dot, no grouping separators or unit text. 0–20 bar is a fixture.
  function parsePressureBar(raw) {
    const text = String(raw).trim();
    if (!/^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$/.test(text)) {
      throw new Error('Enter a decimal number in bar, without grouping separators.');
    }
    const value = Number(text);
    if (!Number.isFinite(value) || value < 0 || value > 20) {
      throw new Error('Pressure must be between 0 and 20 bar.');
    }
    return Object.is(value, -0) ? 0 : value;
  }

  function normalizeIdentifier(raw, { caseInsensitive = true } = {}) {
    const trimmed = String(raw == null ? '' : raw).trim().replace(/\s+/g, ' ');
    if (!trimmed) throw new Error('Enter an identifier.');
    return caseInsensitive ? trimmed.toUpperCase() : trimmed;
  }

  function readingStatus(sample, nowMs, connected) {
    const value = sample ? sample.value : null;
    const finite = typeof value === 'number' && Number.isFinite(value);
    const present = finite && sample.quality !== 'bad';
    const ageMs = nowMs - sample.sampledAt;
    const clockSkew = ageMs < -2000;
    const staleAfter = Number.isFinite(sample.staleAfterMs) ? sample.staleAfterMs : 10000;
    const stale = ageMs > staleAfter;
    let status = 'Current';
    if (!connected) status = 'Disconnected';
    else if (clockSkew) status = 'Clock mismatch';
    else if (stale) status = 'Stale';
    else if (sample.quality === 'uncertain') status = 'Uncertain';
    return {
      status,
      ageMs,
      display: present ? value : 'Unavailable',
      unit: sample.unit || '',
    };
  }

  function formatProgress(done, total) {
    if (total === 0) return { label: 'No checks required', determinate: false, value: null, max: null };
    const known = total !== null && Number.isInteger(total) && total > 0;
    if (!known) return { label: 'Checking requirements…', determinate: false, value: null, max: null };
    const completed = Math.min(total, Math.max(0, Math.trunc(Number.isFinite(done) ? done : 0)));
    return {
      label: `${completed} of ${total} required checks complete`,
      determinate: true,
      value: completed,
      max: total,
    };
  }

  return { commandReducer, commandLabel, parsePressureBar, normalizeIdentifier, readingStatus, formatProgress };
});
