/**
 * Hero rotating word. Pure vanilla — no framer-motion, no React.
 * Reads .rotator-word elements inside .rotator, cycles them
 * every ROTATE_MS with a sliding spring-ish transition.
 */
(function () {
  const ROTATE_MS = 2400;
  const LEAVE_MS = 350;

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".rotator").forEach((rotator) => {
      const words = Array.from(rotator.querySelectorAll(".rotator-word:not(.rotator-sizer)"));
      if (words.length < 2) return;

      let active = 0;
      words[active].classList.add("is-active");

      // Ensure the rotator is at least as wide as the widest word
      // (prevents the heading from wrapping unpredictably mid-rotation).
      const measure = () => {
        let maxW = 0;
        words.forEach((w) => {
          const wasActive = w.classList.contains("is-active");
          if (!wasActive) w.classList.add("is-active");
          maxW = Math.max(maxW, w.getBoundingClientRect().width);
          if (!wasActive) w.classList.remove("is-active");
        });
        rotator.style.minWidth = Math.ceil(maxW) + "px";
      };
      // wait for fonts to load before measuring
      if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(measure);
      } else {
        setTimeout(measure, 200);
      }
      window.addEventListener("resize", measure);

      setInterval(() => {
        const current = words[active];
        current.classList.remove("is-active");
        current.classList.add("is-leaving");
        active = (active + 1) % words.length;
        const next = words[active];
        // tiny stagger so the outgoing word starts moving before the next arrives
        setTimeout(() => next.classList.add("is-active"), 60);
        setTimeout(() => current.classList.remove("is-leaving"), LEAVE_MS + 80);
      }, ROTATE_MS);
    });
  });
})();
