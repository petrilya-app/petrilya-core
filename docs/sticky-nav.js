/* sticky-nav.js — toggles `.is-scrolled` on .nav once the page
 * scrolls past a small threshold. Lets us keep the nav transparent
 * over the hero and switch to a translucent plate when there's
 * content underneath. Throttled with rAF; passive scroll listener. */
(function () {
  const nav = document.querySelector('.nav');
  if (!nav) return;

  const THRESHOLD = 8;
  let raf = 0;

  const update = () => {
    raf = 0;
    const scrolled = (window.scrollY || window.pageYOffset) > THRESHOLD;
    nav.classList.toggle('is-scrolled', scrolled);
  };

  const onScroll = () => {
    if (raf) return;
    raf = requestAnimationFrame(update);
  };

  update();
  window.addEventListener('scroll', onScroll, { passive: true });
})();
