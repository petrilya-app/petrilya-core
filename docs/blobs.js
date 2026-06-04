/**
 * Background colony blobs.
 *
 * Injects ~14 organic SVG blob shapes into a fixed-position layer
 * behind the page. Each blob has its own:
 *   - position (top/left, viewport-relative)
 *   - size
 *   - color (warm cream / yellow / amber / off-white)
 *   - parallax factor (how much it drifts on scroll)
 *   - subtle continuous wobble (CSS @keyframes)
 *
 * The wobble lives on an inner element so the outer element's
 * parallax transform doesn't conflict with it.
 */
(function () {
  // 5 hand-drawn organic blob silhouettes
  const BLOB_SHAPES = [
    "M50 6 C72 6 92 26 92 50 C92 70 78 92 56 94 C32 94 10 78 8 56 C6 32 26 6 50 6 Z",
    "M44 5 C68 4 90 18 92 40 C95 60 86 80 68 88 C48 95 26 90 14 74 C4 58 5 36 18 20 C28 9 35 5 44 5 Z",
    "M30 12 C52 7 76 14 86 32 C94 50 90 68 76 80 C60 92 38 88 22 80 C8 72 4 54 8 36 C12 22 18 14 30 12 Z",
    "M52 5 C66 6 80 14 88 26 C96 38 92 54 88 66 C82 80 68 92 52 92 C36 95 20 86 10 70 C2 54 6 34 18 20 C28 10 40 5 52 5 Z",
    "M48 8 C68 6 88 18 90 38 C94 50 84 60 86 72 C90 84 78 92 60 90 C42 92 22 86 12 70 C4 54 6 36 20 22 C30 12 38 8 48 8 Z",
    "M50 10 C70 4 86 22 90 38 C96 56 82 76 64 86 C42 94 22 84 14 66 C8 50 12 30 26 18 C36 10 42 10 50 10 Z",
  ];

  // Colors from the brand cream-yellow-amber palette
  const COLORS = [
    "#f5e8c8",   // pale cream
    "#f8dc92",   // soft yellow
    "#e8b86a",   // warm amber
    "#fff0d0",   // off-white
    "#f0c84a",   // brand amber
    "#e5a070",   // soft orange
  ];

  // Each blob: { x: left%, y: top in vh, size: px, rot, parallax, shape, color, opacity, blur }
  // 14 blobs distributed roughly through the page (0–500vh)
  const BLOBS = [
    { x: "8%",   y: 18,   size: 110, rot: 12,   parallax: 0.18, shape: 0, color: 0, opacity: 0.42, blur: 14 },
    { x: "82%",  y: 30,   size: 70,  rot: -25,  parallax: 0.32, shape: 1, color: 1, opacity: 0.50, blur: 10 },
    { x: "55%",  y: 65,   size: 50,  rot: 40,   parallax: 0.55, shape: 2, color: 4, opacity: 0.45, blur: 8  },
    { x: "12%",  y: 95,   size: 130, rot: -8,   parallax: 0.22, shape: 3, color: 2, opacity: 0.35, blur: 16 },
    { x: "90%",  y: 115,  size: 90,  rot: 60,   parallax: 0.40, shape: 4, color: 3, opacity: 0.40, blur: 12 },
    { x: "30%",  y: 150,  size: 60,  rot: -18,  parallax: 0.50, shape: 5, color: 0, opacity: 0.55, blur: 9  },
    { x: "70%",  y: 175,  size: 100, rot: 22,   parallax: 0.28, shape: 1, color: 5, opacity: 0.32, blur: 14 },
    { x: "5%",   y: 210,  size: 80,  rot: 90,   parallax: 0.38, shape: 2, color: 1, opacity: 0.42, blur: 11 },
    { x: "62%",  y: 240,  size: 140, rot: -45,  parallax: 0.20, shape: 0, color: 4, opacity: 0.28, blur: 18 },
    { x: "25%",  y: 280,  size: 55,  rot: 5,    parallax: 0.60, shape: 3, color: 3, opacity: 0.50, blur: 8  },
    { x: "85%",  y: 310,  size: 95,  rot: -30,  parallax: 0.30, shape: 4, color: 0, opacity: 0.38, blur: 13 },
    { x: "45%",  y: 350,  size: 75,  rot: 50,   parallax: 0.45, shape: 5, color: 2, opacity: 0.40, blur: 10 },
    { x: "15%",  y: 390,  size: 110, rot: -10,  parallax: 0.25, shape: 1, color: 1, opacity: 0.35, blur: 15 },
    { x: "78%",  y: 430,  size: 65,  rot: 35,   parallax: 0.52, shape: 2, color: 4, opacity: 0.45, blur: 9  },
  ];

  function init() {
    const layer = document.createElement("div");
    layer.className = "blob-bg";
    layer.setAttribute("aria-hidden", "true");

    BLOBS.forEach((b, i) => {
      const wrap = document.createElement("div");
      wrap.className = "blob";
      wrap.style.left = b.x;
      wrap.style.top = b.y + "vh";
      wrap.style.width = b.size + "px";
      wrap.style.height = b.size + "px";
      wrap.style.opacity = b.opacity;
      wrap.style.setProperty("--parallax", b.parallax);

      const inner = document.createElement("div");
      inner.className = "blob-inner";
      inner.style.filter = `blur(${b.blur}px)`;
      inner.style.transform = `rotate(${b.rot}deg)`;
      // each blob gets a unique drift duration + phase to avoid sync
      inner.style.animationDuration = (16 + (i % 6) * 3) + "s";
      inner.style.animationDelay = (-i * 1.4) + "s";

      inner.innerHTML = `
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
          <path d="${BLOB_SHAPES[b.shape]}" fill="${COLORS[b.color]}"/>
        </svg>
      `;
      wrap.appendChild(inner);
      layer.appendChild(wrap);
    });

    document.body.appendChild(layer);

    // Scroll-driven parallax
    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        layer.style.setProperty("--scroll-y", window.scrollY + "px");
        ticking = false;
      });
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
