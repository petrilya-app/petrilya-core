/**
 * Hero download CTA — visual feedback for browser-native downloads.
 *
 * GitHub Releases redirects cross-origin and doesn't expose CORS, so
 * we can't track real download progress in JavaScript. Instead we
 * acknowledge the click with two brief state transitions on the
 * button:
 *
 *   idle         (default)
 *     ↓ click
 *   is-downloading   (~1.8 s — animated icon, 'starting…' label)
 *     ↓
 *   is-done          (~2.0 s — check mark, green tint)
 *     ↓
 *   idle
 *
 * The actual download is fully handled by the browser via the
 * anchor's native href + download attribute — we never preventDefault.
 */
(function () {
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-hero-download]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.classList.contains("is-downloading")
            || btn.classList.contains("is-done")) {
          return;
        }
        btn.classList.add("is-downloading");
        setTimeout(() => {
          btn.classList.remove("is-downloading");
          btn.classList.add("is-done");
        }, 1800);
        setTimeout(() => {
          btn.classList.remove("is-done");
        }, 3800);
      });
    });
  });
})();
