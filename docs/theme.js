/**
 * Light/dark theme toggle. Runs inline-early (head) to prevent FOUC,
 * then attaches the click handler once the DOM is ready.
 */
(function () {
  const STORAGE_KEY = "petrilya-theme";
  const html = document.documentElement;

  function applyTheme(theme) {
    if (theme === "light") {
      html.classList.add("light");
    } else {
      html.classList.remove("light");
    }
  }

  // Run synchronously to avoid a flash of the wrong theme.
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    const preferred = window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
    applyTheme(saved || preferred);
  } catch (_) {
    /* localStorage might be disabled — silently fall back to dark */
  }

  document.addEventListener("DOMContentLoaded", () => {
    const buttons = document.querySelectorAll("[data-theme-toggle]");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const isLight = html.classList.toggle("light");
        try {
          localStorage.setItem(STORAGE_KEY, isLight ? "light" : "dark");
        } catch (_) {}
      });
    });

    // React to OS-level changes only when the user hasn't picked manually.
    if (!localStorage.getItem(STORAGE_KEY)) {
      window
        .matchMedia("(prefers-color-scheme: light)")
        .addEventListener("change", (e) => {
          applyTheme(e.matches ? "light" : "dark");
        });
    }
  });
})();
