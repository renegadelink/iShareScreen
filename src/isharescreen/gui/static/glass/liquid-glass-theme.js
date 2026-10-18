/* Color customization without dependencies. Accepts #rgb or #rrggbb colors. */
(() => {
  const rgb = value => {
    if (typeof value !== 'string' || !/^#(?:[\da-f]{3}|[\da-f]{6})$/i.test(value)) {
      throw new TypeError('Theme colors must be #rgb or #rrggbb hex colors.');
    }
    const hex = value.length === 4 ? [...value.slice(1)].map(c => c + c).join('') : value.slice(1);
    return [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16));
  };
  const mix = (a, b, amount) => a.map((c, i) => Math.round(c * (1 - amount) + b[i] * amount));
  const color = channels => `rgb(${channels.join(', ')})`;
  const alpha = (channels, amount) => `rgba(${channels.join(', ')}, ${amount})`;
  const luminance = channels => channels.map(c => c / 255).map(c => c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4)
    .reduce((sum, c, i) => sum + c * [.2126, .7152, .0722][i], 0);
  function setTheme({ accent = '#127942', background = '#eef3ee', field = true } = {}, root = document.documentElement) {
    // Validate everything before changing the target.
    const a = rgb(accent), b = rgb(background);
    if (typeof field !== 'boolean') throw new TypeError('field must be a boolean.');
    if (!(root instanceof HTMLElement)) throw new TypeError('Theme root must be an HTML element.');
    const white = [255, 255, 255], black = [0, 0, 0];
    let ink = a;
    // Accent text on this light material needs more contrast than the tint does.
    while ((1.05 / (luminance(ink) + .05)) < 4.5) ink = mix(ink, black, .1);
    const lum = luminance(a);
    const onAccent = (lum + .05) / (luminance([16, 36, 26]) + .05) >= 4.5
      ? '#10241a' : (1.05 / (lum + .05) >= 4.5 ? '#ffffff' : '#000000');
    const values = {
      '--accent': color(ink), '--accent-tint': color(a), '--on-accent': onAccent,
      '--accent-soft': alpha(a, .1), '--accent-line': alpha(a, .32),
      '--brand-green': color(a), '--brand-deep': color(mix(ink, black, .2)),
      '--brand-mid': color(mix(a, white, .22)), '--brand-lime': color(mix(a, white, .5)),
      '--bg': color(b), '--field-a': alpha(mix(a, white, .5), .3),
      '--field-b': alpha(mix(a, white, .22), .22), '--field-c': alpha(a, .16),
      '--background-image': field ? `radial-gradient(1100px 700px at -10% -10%, ${alpha(mix(a, white, .5), .3)}, transparent 60%), radial-gradient(900px 640px at 110% 0%, ${alpha(mix(a, white, .22), .22)}, transparent 62%), radial-gradient(1000px 800px at 60% 120%, ${alpha(a, .16)}, transparent 60%), radial-gradient(700px 500px at 30% 60%, rgba(255,255,255,.55), transparent 70%)` : 'none',
    };
    for (const [key, value] of Object.entries(values)) root.style.setProperty(key, value);
    return values;
  }
  window.LiquidGlass = Object.assign(window.LiquidGlass || {}, { setTheme });
})();
