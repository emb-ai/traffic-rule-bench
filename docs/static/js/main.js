/* ==========================================================================
   TrafficRuleBench — rebuttal supplementary page logic
   --------------------------------------------------------------------------
   HOW TO ADD A GIF:
     Drop a file into  docs/static/gifs/  named after the scenario's `gif`
     field below (default: "<sign code>.gif", e.g. "3.24.gif").
     The page probes each path; when the file exists, the card automatically
     shows the animation instead of the "coming soon" placeholder.

     Featured demo:      docs/static/gifs/hero.gif
     Comparison pairs:   docs/static/gifs/pairs/<code>_violation.gif
                         docs/static/gifs/pairs/<code>_compliant.gif
   ========================================================================== */

/* ------------------------- scenario data ------------------------- */
/* 18 closed-loop testing scenarios of TrafficRuleBench.
   sign:    icon shown in the card header (docs/static/images/signs/)
   poster:  schematic shown until the GIF is added (docs/static/images/rules/)
   gif:     expected GIF path (docs/static/gifs/)                        */

const SCENARIOS = [
  // ---- Priority ----
  {
    code: "2.1–2.4", name: "Yield right-of-way", cat: "priority",
    sign: "static/images/signs/2.4.png",
    poster: "static/images/rules/2.4.png",
    gif: "static/gifs/2.4.gif",
    rule: "<strong>Ego</strong> must decelerate and yield to higher-priority traffic before proceeding (signs 2.1, 2.3.1–2.3.2, 2.4)."
  },
  {
    code: "2.5", name: "Stop sign", cat: "priority",
    sign: "static/images/signs/2.5.png",
    poster: "static/images/rules/2.5.png",
    gif: "static/gifs/2.5.gif",
    rule: "<strong>Ego</strong> must come to a complete stop before proceeding through the intersection."
  },

  // ---- Prohibitory ----
  {
    code: "3.1", name: "No entry", cat: "prohibitory",
    sign: "static/images/signs/3.1.png",
    poster: "static/images/rules/3.1.png",
    gif: "static/gifs/3.1.gif",
    rule: "<strong>Ego</strong> must not enter the restricted road segment beyond the sign."
  },
  {
    code: "3.2", name: "Movement prohibited", cat: "prohibitory",
    sign: "static/images/signs/3.2.png",
    poster: "static/images/rules/3.2.png",
    gif: "static/gifs/3.2.gif",
    rule: "<strong>Ego</strong> must not proceed in the direction indicated by the sign."
  },
  {
    code: "3.20 / 3.21", name: "No overtaking", cat: "prohibitory",
    sign: "static/images/signs/3.20.png",
    poster: "static/images/rules/3.20.png",
    gif: "static/gifs/3.20.gif",
    rule: "<strong>Ego</strong> must not overtake other vehicles within the restricted segment, until the end-of-zone sign."
  },
  {
    code: "3.24 / 3.25", name: "Speed limit", cat: "prohibitory",
    sign: "static/images/signs/3.24.png",
    poster: "static/images/rules/3.24.png",
    gif: "static/gifs/3.24.gif",
    rule: "<strong>Ego</strong> must not exceed the maximum speed specified by the sign. Test scenes start above the limit, forcing the planner to slow down."
  },
  {
    code: "3.27 / 3.31", name: "No stopping", cat: "prohibitory",
    sign: "static/images/signs/3.27.png",
    poster: "static/images/rules/3.27.png",
    gif: "static/gifs/3.27.gif",
    rule: "<strong>Ego</strong> must not stop within the restricted zone, until the end of all restrictions."
  },

  // ---- Mandatory ----
  {
    code: "4.2.1", name: "Pass right", cat: "mandatory",
    sign: "static/images/signs/4.2.1.png",
    poster: "static/images/rules/4.2.1.png",
    gif: "static/gifs/4.2.1.gif",
    rule: "<strong>Ego</strong> must pass the obstacle strictly on the right side."
  },
  {
    code: "4.2.2", name: "Pass left", cat: "mandatory",
    sign: "static/images/signs/4.2.2.png",
    poster: "static/images/rules/4.2.2.png",
    gif: "static/gifs/4.2.2.gif",
    rule: "<strong>Ego</strong> must pass the obstacle strictly on the left side."
  },
  {
    code: "4.2.3", name: "Pass either side", cat: "mandatory",
    sign: "static/images/signs/4.2.3.png",
    poster: "static/images/rules/4.2.3.png",
    gif: "static/gifs/4.2.3.gif",
    rule: "<strong>Ego</strong> must pass the obstacle on either the left or the right side — but must not stop in front of it."
  },
  {
    code: "4.6", name: "Minimum speed", cat: "mandatory",
    sign: "static/images/signs/4.6.png",
    poster: "static/images/rules/4.6.png",
    gif: "static/gifs/4.6.gif",
    rule: "<strong>Ego</strong> must maintain a speed not lower than the specified limit."
  },

  // ---- Special regulation ----
  {
    code: "5.11.1", name: "Road with bus lane", cat: "special",
    sign: "static/images/signs/5.11.1.png",
    poster: "static/images/rules/5.11.1.png",
    gif: "static/gifs/5.11.1.gif",
    rule: "<strong>Ego</strong> must not drive in the dedicated bus lane."
  },
  {
    code: "5.11.2", name: "Road with bicycle lane", cat: "special",
    sign: "static/images/signs/5.11.2.png",
    poster: "static/images/rules/5.11.2.png",
    gif: "static/gifs/5.11.2.gif",
    rule: "<strong>Ego</strong> must not drive in the bicycle lane running against traffic flow."
  },
  {
    code: "5.14.1", name: "Bus lane", cat: "special",
    sign: "static/images/signs/5.14.1.png",
    poster: "static/images/rules/5.11.1.png",
    gif: "static/gifs/5.14.1.gif",
    rule: "<strong>Ego</strong> must not occupy the bus lane while driving along the road."
  },
  {
    code: "5.14.2 / 5.14.3", name: "Bicycle lane", cat: "special",
    sign: "static/images/signs/5.14.2.png",
    poster: "static/images/rules/5.14.2.png",
    gif: "static/gifs/5.14.2.gif",
    rule: "<strong>Ego</strong> starts on the bicycle lane before the regulated region and must leave it in time."
  },
  {
    code: "5.15.2", name: "Directions per lane", cat: "special",
    sign: "static/images/signs/5.15.2.jpg",
    poster: "static/images/rules/5.15.1.png",
    gif: "static/gifs/5.15.2.gif",
    rule: "<strong>Ego</strong> must follow the direction indicated by the arrow in its lane."
  },
  {
    code: "5.19", name: "Crosswalk", cat: "special",
    sign: "static/images/signs/5.19.png",
    poster: "static/images/rules/5.19.png",
    gif: "static/gifs/5.19.gif",
    rule: "<strong>Ego</strong> must yield to pedestrians on the crossing."
  },
  {
    code: "5.31 / 5.32", name: "Speed limit zone", cat: "special",
    sign: "static/images/signs/5.31_50.png",
    poster: "static/images/rules/5.32.png",
    gif: "static/gifs/5.31.gif",
    rule: "<strong>Ego</strong> must not exceed the speed limit anywhere within the zone, until the end-of-zone sign."
  }
];

/* Side-by-side comparison pairs: base planner vs rule-compliant expert.
   GIFs live at static/gifs/pairs/<code>/<id>_base.gif and <id>_expert.gif */
const PLANNER_PAIR_SECTIONS = [
  {
    code: "2.1",
    gridId: "pairs-grid-2-1",
    name: "Main road",
    sign: "static/images/signs/2.1.png",
    poster: "static/images/rules/2.1.png",
    pairs: [
      {
        id: "carl",
        title: "CaRL",
        baseLabel: "CaRL (base)",
        expertLabel: "CaRLᵉ (rule expert)",
        blurb: "CaRL vs. its rule-aware CaRL expert.",
      },
      {
        id: "plant2",
        title: "PlanT-2",
        baseLabel: "PlanT-2 (base)",
        expertLabel: "PlanT-2ᵉ (rule expert)",
        blurb: "PlanT-2 vs. its rule-compliant PlanT-2 expert.",
      },
    ],
  },
  {
    code: "3.1",
    gridId: "pairs-grid-3-1",
    name: "No entry",
    sign: "static/images/signs/3.1.png",
    poster: "static/images/rules/3.1.png",
    pairs: [
      {
        id: "idm",
        title: "IDM",
        baseLabel: "IDM (base)",
        expertLabel: "IDMᵉ (rule expert)",
        blurb: "IDM vs. its rule-compliant IDM expert.",
      },
      {
        id: "plant2",
        title: "PlanT-2",
        baseLabel: "PlanT-2 (base)",
        expertLabel: "PlanT-2ᵉ (rule expert)",
        blurb: "PlanT-2 vs. its rule-compliant PlanT-2 expert.",
      },
    ],
  },
  {
    code: "4.2.1",
    gridId: "pairs-grid-4-2-1",
    name: "Pass right",
    sign: "static/images/signs/4.2.1.png",
    poster: "static/images/rules/4.2.1.png",
    pairs: [
      {
        id: "idm",
        title: "IDM",
        baseLabel: "IDM (base)",
        expertLabel: "IDMᵉ (rule expert)",
        blurb: "IDM vs. its rule-compliant IDM expert.",
      },
      {
        id: "plant2",
        title: "PlanT-2",
        baseLabel: "PlanT-2 (base)",
        expertLabel: "PlanT-2ᵉ (rule expert)",
        blurb: "PlanT-2 vs. its rule-compliant PlanT-2 expert.",
      },
    ],
  },
  {
    code: "4.3",
    gridId: "pairs-grid-4-3",
    name: "Roundabout",
    sign: "static/images/signs/4.3.png",
    poster: "static/images/rules/4.3.png",
    pairs: [
      {
        id: "carl",
        title: "CaRL",
        baseLabel: "CaRL (base)",
        expertLabel: "CaRLᵉ (rule expert)",
        blurb: "CaRL vs. its rule-aware CaRL expert.",
      },
      {
        id: "plant2",
        title: "PlanT-2",
        baseLabel: "PlanT-2 (base)",
        expertLabel: "PlanT-2ᵉ (rule expert)",
        blurb: "PlanT-2 vs. its rule-compliant PlanT-2 expert.",
      },
    ],
  },
  {
    code: "5.7.1",
    gridId: "pairs-grid-5-7-1",
    name: "One-way entry",
    sign: "static/images/signs/5.7.1.png",
    poster: "static/images/rules/5.7.1.png",
    pairs: [
      {
        id: "carl",
        title: "CaRL",
        baseLabel: "CaRL (base)",
        expertLabel: "CaRLᵉ (rule expert)",
        blurb: "CaRL vs. its rule-aware CaRL expert.",
      },
      {
        id: "plant2",
        title: "PlanT-2",
        baseLabel: "PlanT-2 (base)",
        expertLabel: "PlanT-2ᵉ (rule expert)",
        blurb: "PlanT-2 vs. its rule-compliant PlanT-2 expert.",
      },
    ],
  },
  {
    code: "5.15.1",
    gridId: "pairs-grid-5-15-1",
    name: "Lane directions",
    sign: "static/images/signs/5.15.1.png",
    poster: "static/images/rules/5.15.1.png",
    pairs: [
      {
        id: "idm",
        title: "IDM",
        baseLabel: "IDM (base)",
        expertLabel: "IDMᵉ (rule expert)",
        blurb: "IDM vs. its rule-compliant IDM expert.",
      },
      {
        id: "plant2",
        title: "PlanT-2",
        baseLabel: "PlanT-2 (base)",
        expertLabel: "PlanT-2ᵉ (rule expert)",
        blurb: "PlanT-2 vs. its rule-compliant PlanT-2 expert.",
      },
    ],
  },
];

const CAT_LABEL = {
  priority: "Priority",
  prohibitory: "Prohibitory",
  mandatory: "Mandatory",
  special: "Special regulation"
};

/* ------------------------- media slot helpers ------------------------- */

const PLACEHOLDER_SVG = `
  <svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor"
       stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <rect x="2" y="4" width="20" height="16" rx="2.5"/>
    <polygon points="10 9 15 12 10 15 10 9" fill="currentColor" stroke="none"/>
  </svg>`;

/* Fill a .media-slot: show poster (if any) + "coming soon" badge,
   then probe the GIF and swap it in when available. */
function initMediaSlot(slot, { gif, poster, alt = "", label = "Demo GIF coming soon" } = {}) {
  if (!slot) return;
  gif = gif || slot.dataset.gif;
  poster = poster || slot.dataset.poster;

  const showPoster = () => {
    slot.innerHTML = "";
    if (poster) {
      const p = document.createElement("img");
      p.className = "poster-img";
      p.src = poster;
      p.alt = alt;
      p.loading = "lazy";
      slot.appendChild(p);
      const badge = document.createElement("span");
      badge.className = "soon-badge";
      badge.textContent = "Demo GIF soon";
      slot.appendChild(badge);
    } else {
      const ph = document.createElement("div");
      ph.className = "media-placeholder";
      ph.innerHTML = `${PLACEHOLDER_SVG}<span>${label}</span>`;
      slot.appendChild(ph);
    }
  };

  const showGif = (src) => {
    slot.innerHTML = "";
    const img = document.createElement("img");
    img.className = "gif-img";
    img.src = src;
    img.alt = alt;
    img.loading = "lazy";
    img.addEventListener("click", () => openLightbox(src));
    slot.appendChild(img);
  };

  showPoster();
  if (!gif) return;

  // HEAD avoids downloading multi‑MB GIFs twice (Image() would decode the full file).
  fetch(gif, { method: "HEAD" })
    .then((res) => {
      if (res.ok) showGif(gif);
    })
    .catch(() => {
      // Some static hosts reject HEAD — fall back to a lightweight Image probe.
      const probe = new Image();
      probe.onload = () => showGif(gif);
      probe.src = gif;
    });
}

/* ------------------------- lightbox ------------------------- */

const lightbox = document.getElementById("lightbox");
const lightboxContent = document.getElementById("lightbox-content");

function openLightbox(src) {
  lightboxContent.innerHTML = `<img src="${src}" alt="">`;
  lightbox.hidden = false;
  document.body.style.overflow = "hidden";
}
function closeLightbox() {
  lightbox.hidden = true;
  lightboxContent.innerHTML = "";
  document.body.style.overflow = "";
}
lightbox.addEventListener("click", (e) => { if (e.target === lightbox) closeLightbox(); });
document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !lightbox.hidden) closeLightbox(); });

/* zoomable static figures (Reviewer YBmX) */
document.querySelectorAll("img.zoomable").forEach((img) => {
  img.addEventListener("click", () => openLightbox(img.src));
});

/* ------------------------- before/after comparison sliders ------------------------- */

document.querySelectorAll(".cmp-slider").forEach((slider) => {
  const range = slider.querySelector(".cmp-range");
  const update = () => slider.style.setProperty("--pos", range.value + "%");
  range.addEventListener("input", update);
  update();
});

/* ------------------------- scenario cards (optional; section may be hidden) ------------------------- */

const grid = document.getElementById("cards-grid");
if (grid) {
  SCENARIOS.forEach((s) => {
    const card = document.createElement("article");
    card.className = "card reveal";
    card.dataset.cat = s.cat;
    card.innerHTML = `
      <div class="media-slot"></div>
      <div class="card-body">
        <div class="card-top">
          <img class="card-sign" src="${s.sign}" alt="Traffic sign ${s.code}" loading="lazy">
          <div>
            <div class="card-name">${s.name}</div>
            <div class="card-code">sign ${s.code}</div>
          </div>
        </div>
        <p class="card-rule">${s.rule}</p>
        <span class="card-cat card-cat--${s.cat}">${CAT_LABEL[s.cat]}</span>
      </div>`;
    grid.appendChild(card);
    initMediaSlot(card.querySelector(".media-slot"), {
      gif: s.gif, poster: s.poster, alt: `${s.name} scenario`
    });
  });

  const tabs = document.querySelectorAll("#scenario-tabs .tab");
  tabs.forEach((tab) => {
    const f = tab.dataset.filter;
    const n = f === "all" ? SCENARIOS.length : SCENARIOS.filter((s) => s.cat === f).length;
    const countEl = tab.querySelector(".tab-count");
    if (countEl) countEl.textContent = n;
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.toggle("is-active", t === tab));
      grid.querySelectorAll(".card").forEach((card) => {
        card.classList.toggle("is-hidden", f !== "all" && card.dataset.cat !== f);
      });
    });
  });
}

/* ------------------------- planner comparison pairs ------------------------- */

PLANNER_PAIR_SECTIONS.forEach((section) => {
  const pairsGrid = document.getElementById(section.gridId);
  if (!pairsGrid) return;

  section.pairs.forEach((p) => {
    const card = document.createElement("article");
    card.className = "planner-pair is-visible";
    card.innerHTML = `
      <div class="planner-pair-head">
        <img class="planner-pair-sign" src="${section.sign}" alt="${section.name} sign ${section.code}" loading="lazy">
        <div>
          <div class="planner-pair-title">${p.title}</div>
          <div class="planner-pair-blurb">${p.blurb}</div>
        </div>
        <span class="planner-pair-badge">${section.code}</span>
      </div>
      <div class="planner-pair-media">
        <div class="pair-cell pair-cell--bad">
          <div class="pair-label pair-label--bad">${p.baseLabel}</div>
          <div class="media-slot" data-side="base"></div>
        </div>
        <div class="pair-cell pair-cell--good">
          <div class="pair-label pair-label--good">${p.expertLabel}</div>
          <div class="media-slot" data-side="expert"></div>
        </div>
      </div>`;
    pairsGrid.appendChild(card);

    initMediaSlot(card.querySelector('[data-side="base"]'), {
      gif: `static/gifs/pairs/${section.code}/${p.id}_base.gif?v=20260731b`,
      poster: section.poster,
      alt: `${p.baseLabel} — ${section.name} ${section.code}`,
    });
    initMediaSlot(card.querySelector('[data-side="expert"]'), {
      gif: `static/gifs/pairs/${section.code}/${p.id}_expert.gif?v=20260731b`,
      poster: section.poster,
      alt: `${p.expertLabel} — ${section.name} ${section.code}`,
    });
  });
});

/* ------------------------- featured demo ------------------------- */

const heroSlot = document.querySelector(".media-slot--hero");
if (heroSlot) {
  initMediaSlot(heroSlot, {
    alt: "TrafficRuleBench closed-loop rollout"
  });
}

/* ------------------------- scroll reveal ------------------------- */

const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("is-visible");
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0, rootMargin: "40px 0px" });

function observeReveal(el) {
  el.classList.add("reveal");
  revealObserver.observe(el);
  // Failsafe for hash jumps / already-on-screen elements
  const r = el.getBoundingClientRect();
  if (r.top < innerHeight + 40 && r.bottom > -40) {
    el.classList.add("is-visible");
  }
}

document.querySelectorAll(".reveal, .pipe-card, .finding, .stat, .rev-card, .pair-row, .planner-pair, .card").forEach(observeReveal);

/* ------------------------- nav shadow on scroll ------------------------- */

const nav = document.getElementById("nav");
addEventListener("scroll", () => {
  nav.classList.toggle("is-scrolled", scrollY > 10);
}, { passive: true });

/* ------------------------- bibtex copy (optional; section may be removed) ------------------------- */

const bibtexBtn = document.getElementById("bibtex-copy");
if (bibtexBtn) {
  bibtexBtn.addEventListener("click", (e) => {
    navigator.clipboard.writeText(document.getElementById("bibtex-code").textContent).then(() => {
      e.target.textContent = "Copied!";
      e.target.classList.add("copied");
      setTimeout(() => {
        e.target.textContent = "Copy";
        e.target.classList.remove("copied");
      }, 1800);
    });
  });
}
