/**
 * Layered Petri dish hero composition.
 *
 * Composes the dish image from real PNG components:
 *   - agar.png as the base (the dish + agar already photographed
 *     together by the user)
 *   - dozens of colony PNG instances scattered across the agar,
 *     placed with a Poisson-like non-overlap algorithm with a
 *     deterministic seed so the layout is stable across reloads
 *
 * The number, size distribution, type mix (yellow vs cream) and
 * placement radius can be tuned per-instance with data-* attrs.
 */
(function () {
  // ---- seeded RNG so the layout is stable but feels random ----
  function rng(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  // Available colony assets, relative to the page that loads this script.
  // The hero stage element's data-asset-base attribute overrides the path
  // prefix so /ru/ can point at "../" instead of "".
  const TYPES = ["yellow", "cream"];

  function place(stage) {
    const seed     = parseInt(stage.dataset.seed     || "42",  10);
    const count    = parseInt(stage.dataset.count    || "40",  10);
    const yellowMix= parseFloat(stage.dataset.yellow || "0.6");
    const base     = stage.dataset.assetBase || "";
    const random   = rng(seed);

    // Placement radius — fraction of stage half-width. The agar disc in
    // agar.png occupies ~72% of the PNG canvas (the rest is plastic rim
    // and transparent padding), so we use 0.30 here to keep every colony
    // comfortably INSIDE the cream agar area rather than crowding the rim.
    const R = 0.30;
    // colony size distribution — also slightly tighter since stage is smaller
    const SIZE_SMALL  = [0.035, 0.060];  // 3.5%-6% of stage width
    const SIZE_MED    = [0.060, 0.085];
    const SIZE_LARGE  = [0.085, 0.115];
    // min separation as fraction of (sizeA + sizeB)
    const MIN_GAP     = 0.60;

    const placed = [];
    let attempts = 0;
    const maxAttempts = count * 80;

    while (placed.length < count && attempts < maxAttempts) {
      attempts++;
      // uniform sample inside a disc
      const angle = random() * Math.PI * 2;
      const r = R * Math.sqrt(random());
      const x = 0.5 + Math.cos(angle) * r;
      const y = 0.5 + Math.sin(angle) * r;

      // size bucket: 50% small, 35% medium, 15% large
      const sPick = random();
      const range = sPick < 0.5 ? SIZE_SMALL
                  : sPick < 0.85 ? SIZE_MED
                  : SIZE_LARGE;
      const size = range[0] + random() * (range[1] - range[0]);

      // check non-overlap with previously placed
      let collides = false;
      for (let i = 0; i < placed.length; i++) {
        const p = placed[i];
        const dx = p.x - x, dy = p.y - y;
        const minDist = (p.size + size) * MIN_GAP;
        if (dx * dx + dy * dy < minDist * minDist) {
          collides = true;
          break;
        }
      }
      if (collides) continue;

      const type = random() < yellowMix ? "yellow" : "cream";
      const rotation = Math.floor(random() * 360);
      placed.push({ x, y, size, type, rotation });
    }

    // sort top → bottom for natural z-order (lower colonies drawn last)
    placed.sort((a, b) => a.y - b.y);

    const frag = document.createDocumentFragment();
    placed.forEach((c, i) => {
      const img = document.createElement("img");
      img.src = base + "colony-" + c.type + "-m-smooth.png";
      img.alt = "";
      img.className = "colony";
      img.loading = "lazy";
      img.decoding = "async";
      img.style.left   = (c.x * 100) + "%";
      img.style.top    = (c.y * 100) + "%";
      img.style.width  = (c.size * 100) + "%";
      img.style.transform = "translate(-50%, -50%) rotate(" + c.rotation + "deg)";
      img.style.animationDelay = (i * 35) + "ms";
      img.style.zIndex = "10";
      frag.appendChild(img);
    });
    stage.appendChild(frag);
    stage.dataset.placed = placed.length;
  }

  function init() {
    document.querySelectorAll(".dish-stage").forEach(place);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
