/* =====================================================================
 * steps-carousel.js
 * On narrow viewports (.steps) and the features grid (.features .grid)
 * become horizontal scroll-snap carousels (see style.css). This script:
 *   - injects dot indicators under each carousel
 *   - syncs the active dot as the user swipes
 *   - jumps to the matching card on dot click
 *
 * No-ops on desktop (>768px) because the dots are display:none above
 * 768px in CSS. The DOM is still built so resizing the window picks up
 * the right state without a reload.
 * ===================================================================== */
(function () {
  const MQ = window.matchMedia('(max-width: 768px)');

  // selector → which child elements count as "slides"
  const TARGETS = [
    { container: '.steps',         slide: '.step',     label: 'Шаги' },
    { container: '.features .grid', slide: '.card',    label: 'Возможности' },
  ];

  TARGETS.forEach(({ container, slide, label }) => {
    document.querySelectorAll(container).forEach((track) => initCarousel(track, slide, label));
  });

  function initCarousel(track, slideSel, label) {
    const cards = track.querySelectorAll(slideSel);
    if (cards.length < 2) return;

    // Build dot row
    const dots = document.createElement('div');
    dots.className = 'steps-dots';
    dots.setAttribute('role', 'tablist');
    dots.setAttribute('aria-label', label);

    cards.forEach((_, i) => {
      const d = document.createElement('button');
      d.type = 'button';
      d.className = 'steps-dot';
      d.setAttribute('role', 'tab');
      d.setAttribute('aria-label', `${label}: к слайду ${i + 1}`);
      if (i === 0) d.classList.add('is-active');
      d.addEventListener('click', () => {
        cards[i].scrollIntoView({ behavior: 'smooth', inline: 'start', block: 'nearest' });
      });
      dots.appendChild(d);
    });

    track.parentNode.insertBefore(dots, track.nextSibling);
    const dotEls = dots.querySelectorAll('.steps-dot');

    // Sync active dot with scroll position. Throttled with rAF.
    let raf = 0;
    const onScroll = () => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        if (!MQ.matches) return;
        const r = track.getBoundingClientRect();
        const centerX = r.left + r.width / 2;
        let bestIdx = 0;
        let bestDist = Infinity;
        cards.forEach((c, i) => {
          const cr = c.getBoundingClientRect();
          const cx = cr.left + cr.width / 2;
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
    track.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
  }
})();
