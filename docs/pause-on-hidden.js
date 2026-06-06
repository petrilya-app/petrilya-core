/**
 * Tags <html> with `is-hidden` whenever the tab is in the background
 * (visibilitychange API). CSS rules under html.is-hidden then pause
 * every long-running animation: the 14 background colony blobs, the
 * floating hero dish, the rotating dashed rings, the slot-machine
 * word rotator timer, etc. Result: a backgrounded petrilya.com tab
 * drops from ~10-15% CPU on a mid-range laptop to ~0%, saving
 * battery and respecting the user's other open work.
 */
(function () {
  const root = document.documentElement;
  function sync() {
    root.classList.toggle("is-hidden", document.hidden);
  }
  sync();
  document.addEventListener("visibilitychange", sync);
})();
