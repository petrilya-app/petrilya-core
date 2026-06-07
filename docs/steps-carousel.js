/* =====================================================================
 * steps-carousel.js
 * On narrow viewports the .steps row becomes a horizontal scroll-snap
 * carousel (see style.css). This script adds:
 *   - dot indicators under the carousel
 *   - active-dot sync as the user swipes
 *   - click-on-dot jumps to the matching step
 *
 * No-ops on desktop (>768px) because the CSS leaves .steps as a grid
 * there. We still build the dots — they're display:none above 768px —
 * so resizing the window picks up the right state without a reload.
 * ===================================================================== */
(function () {
  const MQ = window.matchMedia('(max-width: 768px)');

  document.querySelectorAll('.steps').forEach((steps) => {
    const cards = steps.querySelectorAll('.step');
    if (cards.length < 2) return;

    // Build dot row
    const dots = document.createElement('div');
    dots.className = 'steps-dots';
    dots.setAttribute('role', 'tablist');
    dots.setAttribute('aria-label', 'Шаги');

    cards.forEach((_, i) => {
      const d = document.createElement('button');
      d.type = 'button';
      d.className = 'steps-dot';
      d.setAttribute('role', 'tab');
      d.setAttribute('aria-label', `Перейти к шагу ${i + 1}`);
      if (i === 0) d.classList.add('is-active');
      d.addEventListener('click', () => {
        cards[i].scrollIntoView({ behavior: 'smooth', inline: 'start', block: 'nearest' });
      });
      dots.appendChild(d);
    });

    steps.parentNode.insertBefore(dots, steps.nextSibling);
    const dotEls = dots.querySelectorAll('.steps-dot');

    // Sync active dot with scroll position. Throttled with rAF.
    let raf = 0;
    const onScroll = () => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        if (!MQ.matches) return;
        const stepsRect = steps.getBoundingClientRect();
        const centerX = stepsRect.left + stepsRect.width / 2;
        let bestIdx = 0;
        let bestDist = Infinity;
        cards.forEach((c, i) => {
          const r = c.getBoundingClientRect();
          const cx = r.left + r.width / 2;
          const d = Math.abs(cx - centerX);
          if (d < bestDist) {
            bestDist = d;
            bestIdx = i;
          }
        });
        dotEls.forEach((el, i) =>
          el.classList.toggle('is-active', i === bestIdx)
        );
      });
    };
    steps.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
  });
})();
