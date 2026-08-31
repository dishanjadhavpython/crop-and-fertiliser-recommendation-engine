/* Regur — client for the crop & fertiliser engine. */
const $ = (s, r = document) => r.querySelector(s);
const api = (p, o) => fetch(p, o).then(r => r.ok ? r.json()
  : r.json().then(e => Promise.reject(new Error(e.detail || r.statusText))));

const state = { talukas: [], picked: null, season: null, irrigated: false, data: null, sel: 0 };
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ── boot ────────────────────────────────────────────────────────────── */
(async function boot() {
  const [t, s, atlas] = await Promise.all([
    api("/talukas"), api("/seasons"), api("/atlas")]);
  state.talukas = t.talukas;
  state.atlas = atlas;
  drawAtlas(atlas);
  $("#seasons").innerHTML = s.seasons.map((x, i) =>
    `<button type="button" role="radio" aria-checked="${i === 0}" class="${i === 0 ? "on" : ""}"
      data-season="${x}">${x}</button>`).join("");
  state.season = s.seasons[0];
  $("#seasons").onclick = e => {
    const b = e.target.closest("button"); if (!b) return;
    $$("#seasons button").forEach(x => { x.classList.remove("on"); x.setAttribute("aria-checked", "false"); });
    b.classList.add("on"); b.setAttribute("aria-checked", "true"); state.season = b.dataset.season;
  };
  $(".toggle").onclick = e => {
    const b = e.target.closest("button"); if (!b) return;
    $$(".toggle button").forEach(x => { x.classList.remove("on"); x.setAttribute("aria-checked", "false"); });
    b.classList.add("on"); b.setAttribute("aria-checked", "true");
    state.irrigated = b.dataset.irr === "true";
  };
  // only once every control can respond to a click
  restoreFromURL();
})();
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

/* ── the atlas ────────────────────────────────────────────────────────────
   Maharashtra is drawn from the feature store itself: one dot per taluka at
   its real coordinates, shaded by aridity. Nothing here is a stock outline —
   the shape of the state emerges from where the 351 surveyed talukas are. */
/* Maharashtra spans ~7.8 deg of longitude and ~6.1 of latitude; after the
   cos(lat) correction that is 7.4 x 6.1, so the frame is wider than tall.
   A portrait viewBox squashes the state into something unrecognisable. */
const ATLAS = { w: 620, h: 500, pad: 26 };

/* The supplied district map is a plain equirectangular plate, and its aspect
   (1.208) matches Maharashtra's true lat-corrected aspect (1.214) closely
   enough to georeference directly. STATE is the real extent of the state;
   INSET is where the outline sits inside the image, which carries whitespace
   on every side. Both are needed or the dots float off the coast. */
const STATE = { lon: [72.65, 80.90], lat: [15.60, 22.03] };
const INSET = { x0: 0.055, x1: 0.970, y0: 0.040, y1: 0.960 };

function projector(bounds) {
  // project against the *state* extent, not the taluka centroids' bounding
  // box — centroids sit inside the border, so fitting to them would stretch
  // the dots past the coastline
  const [lat0, lat1] = STATE.lat, [lon0, lon1] = STATE.lon;
  // longitude degrees shrink with latitude; without the correction the state
  // comes out visibly too wide
  const k = Math.cos((((lat0 + lat1) / 2) * Math.PI) / 180);
  const spanX = (lon1 - lon0) * k, spanY = lat1 - lat0;
  const scale = Math.min((ATLAS.w - ATLAS.pad * 2) / spanX,
                         (ATLAS.h - ATLAS.pad * 2) / spanY);
  const offX = (ATLAS.w - spanX * scale) / 2, offY = (ATLAS.h - spanY * scale) / 2;
  return (lon, lat) => [
    offX + (lon - lon0) * k * scale,
    ATLAS.h - offY - (lat - lat0) * scale,
  ];
}

/* arid -> wet, matching the legend swatch */
function aridColour(a, lo, hi) {
  // arid -> wet across the supplied palette: vermilion, sand, cream
  const stops = [[227,83,54],[244,164,96],[245,245,220]];
  const t = Math.max(0, Math.min(1, (a - lo) / (hi - lo || 1))) * (stops.length - 1);
  const i = Math.min(Math.floor(t), stops.length - 2), f = t - i;
  const c = stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * f));
  return `rgb(${c.join(",")})`;
}

function drawAtlas(atlas) {
  const svg = $("#atlas");
  const project = projector(atlas.bounds);
  const [lo, hi] = atlas.aridity_range;
  state.project = project;

  // place the plate so its outline lands exactly on the projected extent
  const [ax, ay] = project(STATE.lon[0], STATE.lat[1]);   // north-west
  const [bx, by] = project(STATE.lon[1], STATE.lat[0]);   // south-east
  const outlineW = bx - ax, outlineH = by - ay;
  const imgW = outlineW / (INSET.x1 - INSET.x0);
  const imgH = outlineH / (INSET.y1 - INSET.y0);
  const imgX = ax - INSET.x0 * imgW, imgY = ay - INSET.y0 * imgH;
  const plate = `<image class="plate" href="/static/maharashtra.png"
      x="${imgX.toFixed(1)}" y="${imgY.toFixed(1)}"
      width="${imgW.toFixed(1)}" height="${imgH.toFixed(1)}"
      preserveAspectRatio="none"/>`;

  atlas.talukas.forEach(k => { const [x, y] = project(k.x, k.y); k._x = x; k._y = y; });
  const cx = ATLAS.w / 2, cy = ATLAS.h / 2;
  const far = Math.max(...atlas.talukas.map(k => Math.hypot(k._x - cx, k._y - cy))) || 1;

  const dots = atlas.talukas.map(k => {
    // seeded outward from the centre of the state, so the map grows into
    // being rather than appearing all at once
    const delay = 420 + (Math.hypot(k._x - cx, k._y - cy) / far) * 620;
    return `<circle class="tk" cx="${k._x.toFixed(1)}" cy="${k._y.toFixed(1)}" r="0"
      fill="${aridColour(k.a, lo, hi)}"
      data-d="${k.d}" data-t="${k.t}"
      style="animation:seed .7s var(--ease) ${delay.toFixed(0)}ms backwards"><title>${
        title(k.t)}, ${title(k.d)} — ${k.r} mm/yr</title></circle>`;
  }).join("");

  svg.innerHTML = plate + dots +
    `<circle class="focusring" cx="0" cy="0" r="9"></circle>
     <circle class="halo" cx="0" cy="0" r="6"></circle>
     <circle class="pin" cx="0" cy="0" r="5.5"></circle>
     <text class="lbl" x="0" y="0"></text>`;

  $("#ar-lo").textContent = lo.toFixed(1);
  $("#ar-hi").textContent = hi.toFixed(1);

  requestAnimationFrame(() => $$("#atlas .tk").forEach(c => c.setAttribute("r", 3.9)));
  document.body.classList.add("ready");

  // A dot-per-region map is unusable without a keyboard path to each region.
  // Arrow keys walk to the geographically nearest taluka in that direction,
  // which is how someone reading a map expects to move.
  svg.setAttribute("tabindex", "0");
  let cursorTk = null;

  svg.addEventListener("keydown", e => {
    const dirs = { ArrowLeft: [-1, 0], ArrowRight: [1, 0],
                   ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (e.key === "Enter" || e.key === " ") {
      if (cursorTk) {
        e.preventDefault();
        const hit = state.talukas.find(t =>
          t.District === cursorTk.d && t.Taluka === cursorTk.t);
        if (hit) { pick(hit); run(); }
      }
      return;
    }
    const dir = dirs[e.key];
    if (!dir) return;
    e.preventDefault();
    const from = cursorTk || state.atlas.talukas[0];
    const next = nearestIn(from, dir) || from;
    cursorTk = next;
    const ring = $("#atlas .focusring");
    ring.setAttribute("cx", next._x); ring.setAttribute("cy", next._y);
    ring.classList.add("on");
    $("#atlas-read").textContent =
      `${title(next.t)} · ${title(next.d)} · ${next.r} mm/yr`;
  });

  svg.addEventListener("blur", () => $("#atlas .focusring").classList.remove("on"));

  svg.addEventListener("click", e => {
    const c = e.target.closest(".tk"); if (!c) return;
    const hit = state.talukas.find(t =>
      t.District === c.dataset.d && t.Taluka === c.dataset.t);
    if (hit) { pick(hit); run(); }
  });
  svg.addEventListener("mousemove", e => {
    const c = e.target.closest(".tk");
    $("#atlas-read").textContent = c
      ? `${title(c.dataset.t)} · ${title(c.dataset.d)}`
      : "351 talukas · shaded by aridity";
  });
}

/* nearest taluka in a compass direction, weighted so the walk stays on-axis */
function nearestIn(from, [dx, dy]) {
  let best = null, bestCost = Infinity;
  for (const k of state.atlas.talukas) {
    if (k === from) continue;
    const vx = k._x - from._x, vy = k._y - from._y;
    const along = vx * dx + vy * dy;
    if (along <= 2) continue;                       // wrong side
    const off = Math.abs(vx * dy - vy * dx);        // perpendicular drift
    const cost = along + off * 2.5;
    if (cost < bestCost) { bestCost = cost; best = k; }
  }
  return best;
}

function markAtlas(district, taluka) {
  if (!state.atlas) return;
  const k = state.atlas.talukas.find(x => x.d === district && x.t === taluka);
  if (!k) return;
  for (const sel of [".halo", ".pin"]) {
    const el = $("#atlas " + sel);
    el.setAttribute("cx", k._x); el.setAttribute("cy", k._y);
    el.classList.add("on");
  }
  const lbl = $("#atlas .lbl");
  const flip = k._x > ATLAS.w * 0.62;
  lbl.setAttribute("x", k._x + (flip ? -15 : 15));
  lbl.setAttribute("y", k._y - 9);
  lbl.setAttribute("text-anchor", flip ? "end" : "start");
  lbl.textContent = title(k.t);
  lbl.classList.add("on");
  $$("#atlas .tk").forEach(c => c.classList.add("dim"));
  $("#atlas-read").textContent =
    `${k.r} mm/yr · ${k.w} mm root-zone water · ${k.l} day growing period`;
}

/* ── taluka combobox ─────────────────────────────────────────────────── */
const input = $("#taluka"), list = $("#taluka-list"), hint = $("#district-hint");
let cursor = -1;

function title(s) { return s.replace(/\w\S*/g, w => w[0] + w.slice(1).toLowerCase()); }

function openList(q) {
  const needle = q.trim().toLowerCase();
  const hits = (needle ? state.talukas.filter(t =>
    t.Taluka.toLowerCase().includes(needle) || t.District.toLowerCase().includes(needle)
  ) : state.talukas).slice(0, 60);
  if (!hits.length) { closeList(); return; }
  list.innerHTML = hits.map((t, i) =>
    `<li role="option" data-i="${i}" aria-selected="false">
       <span>${title(t.Taluka)}</span><small>${title(t.District)}</small></li>`).join("");
  list.hidden = false; input.setAttribute("aria-expanded", "true");
  cursor = -1; list._hits = hits;
}
function closeList() { list.hidden = true; input.setAttribute("aria-expanded", "false"); cursor = -1; }
function pick(t) {
  state.picked = t; input.value = title(t.Taluka);
  hint.textContent = `${title(t.District)} district`;
  closeList();
  markAtlas(t.District, t.Taluka);
}

input.addEventListener("input", () => { state.picked = null; hint.textContent = " "; openList(input.value); });
input.addEventListener("focus", () => openList(input.value));
input.addEventListener("keydown", e => {
  const items = $$("li", list);
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault(); if (!items.length) return;
    cursor = e.key === "ArrowDown" ? Math.min(cursor + 1, items.length - 1) : Math.max(cursor - 1, 0);
    items.forEach((li, i) => li.setAttribute("aria-selected", i === cursor));
    items[cursor].scrollIntoView({ block: "nearest" });
  } else if (e.key === "Enter" && cursor >= 0) { e.preventDefault(); pick(list._hits[cursor]); }
  else if (e.key === "Escape") closeList();
});
list.addEventListener("mousedown", e => {
  const li = e.target.closest("li"); if (li) { e.preventDefault(); pick(list._hits[+li.dataset.i]); }
});
document.addEventListener("click", e => { if (!e.target.closest(".combo")) closeList(); });

/* ── deep links ──────────────────────────────────────────────────────── */
/* A recommendation is a thing people forward to each other, so the query
   lives in the URL and a pasted link reopens the same answer. */
function restoreFromURL() {
  const q = new URLSearchParams(location.search);
  const taluka = q.get("taluka"), district = q.get("district");
  if (!taluka) return;
  const hit = state.talukas.find(t =>
    t.Taluka.toLowerCase() === taluka.toLowerCase() &&
    (!district || t.District.toLowerCase() === district.toLowerCase()));
  if (!hit) return;
  pick(hit);

  const season = q.get("season");
  if (season) {
    const b = $(`#seasons button[data-season="${CSS.escape(season)}"]`);
    if (b) b.click();
  }
  if (q.get("irrigated") === "true") $('.toggle button[data-irr="true"]').click();
  for (const [id, key] of [["n", "n"], ["p", "p"], ["k", "k"], ["oc", "oc"]]) {
    const v = q.get(key); if (v) $("#" + id).value = v;
  }
  run();
}

function writeURL() {
  const q = new URLSearchParams({
    district: state.picked.District, taluka: state.picked.Taluka,
    season: state.season,
  });
  if (state.irrigated) q.set("irrigated", "true");
  history.replaceState(null, "", "?" + q);
}

/* ── run ─────────────────────────────────────────────────────────────── */
$("#query").addEventListener("submit", e => { e.preventDefault(); run(); });

async function run() {
  if (!state.picked) { input.focus(); hint.textContent = "Pick a taluka from the list"; return; }
  writeURL();

  const soil = {};
  for (const [id, key] of [["n", "n_kg_ha"], ["p", "p_kg_ha"], ["k", "k_kg_ha"], ["oc", "oc_pct"]]) {
    const v = $("#" + id).value; if (v !== "") soil[key] = parseFloat(v);
  }

  $("#empty").hidden = true; $("#results").hidden = true; $("#loading").hidden = false;
  $(".run").disabled = true;

  try {
    const data = await api("/recommend", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        district: state.picked.District, taluka: state.picked.Taluka,
        season: state.season, irrigated: state.irrigated, top_k: 5,
        soil_test: Object.keys(soil).length ? soil : null,
      }),
    });
    state.data = data; state.sel = 0;
    render(data);
  } catch (err) {
    $("#loading").hidden = true; $("#empty").hidden = false;
    $("#empty p").textContent = err.message;
  } finally { $(".run").disabled = false; }
}

/* ── reveal on scroll, and count the context numbers up ───────────────── */
const revealer = "IntersectionObserver" in window && !reduced
  ? new IntersectionObserver((entries, obs) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        e.target.classList.add("seen");
        obs.unobserve(e.target);           // reveal once, never re-trigger
        if (e.target.dataset.count) {
          e.target.textContent = fmtNum(0, e.target.dataset.dp | 0);
          countUp(e.target);
        }
      }
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 })
  : null;

function reveal(el, delay = 0) {
  if (!revealer) { el.classList.add("seen"); return; }
  el.classList.add("reveal");
  el.style.transitionDelay = `${delay}ms`;
  revealer.observe(el);
  // Hard floor. Anything hidden by an effect must reappear even if the
  // observer never fires — a missed callback should cost an animation, not
  // the content. Without this the whole context strip can stay invisible.
  setTimeout(() => {
    if (!el.classList.contains("seen")) {
      el.classList.add("seen");
      revealer.unobserve(el);
      el.querySelectorAll?.("[data-count]").forEach(n => countUp(n));
    }
  }, 1400);
}

/* A number that ticks up reads as measured rather than decorative — but only
   for the context strip, where every value is a real quantity. */
function fmtNum(n, dp) {
  return dp ? Number(n).toFixed(dp) : Math.round(n).toLocaleString("en-IN");
}

function countUp(el) {
  const target = parseFloat(el.dataset.count);
  if (!isFinite(target)) return;
  const dp = el.dataset.dp | 0, dur = 620, t0 = performance.now();
  (function tick(now) {
    const prog = Math.min(1, (now - t0) / dur);
    if (prog < 1) {
      el.textContent = fmtNum(target * (1 - Math.pow(1 - prog, 3)), dp);
      requestAnimationFrame(tick);
    } else {
      el.textContent = fmtNum(target, dp);   // always lands exactly on truth
    }
  })(t0);
}

/* ── render ──────────────────────────────────────────────────────────── */
function render(d) {
  $("#loading").hidden = true; $("#results").hidden = false;

  const c = d.context;
  const cards = [
    ["Rainfall", "<u>mm/yr</u>", c.annual_rainfall_mm],
    ["Root-zone water", "<u>mm</u>", c.rootzone_awc_mm],
    ["Aridity", "", c.aridity_index, 2],
    ["Growing period", "<u>days</u>", c.lgp_days],
    ["Soil fertility", d.soil_class],
    ["SHC samples", "", c.shc_samples],
  ].map(([k, v, n, dp]) => {
    // The true value goes into the markup; a placeholder never does. A
    // count-up starting at 0 shows "0 mm/yr" for a 742 mm taluka if the
    // observer or rAF never fires — that is wrong data, not a missing
    // flourish, and no animation is worth that.
    const shown = n != null ? fmtNum(n, dp || 0) : v;
    const attrs = n != null ? ` data-count="${n}" data-dp="${dp || 0}"` : "";
    return `<div class="ctx"><b>${k}</b><span><u class="num"${attrs}>${shown}</u>${
      n != null ? v : ""}</span></div>`;
  });

  if (d.water_limited) cards.push(`<div class="ctx flag"><b>Water</b>
    <span>Most crops viable here need irrigation this season.</span></div>`);
  if (!d.confident) cards.push(`<div class="ctx flag"><b>Confidence</b>
    <span>Outside the training distribution — rules are carrying this.</span></div>`);
  $("#context").innerHTML = cards.join("");

  $("#shortlist-note").textContent =
    `${d.crops.length} of ${d.crops.length + d.vetoed.length} candidates cleared the gate`;

  // The arc borrows the reference composition, but height follows *rank*
  // rather than position: the top recommendation sits highest, so visual
  // weight and meaning point the same way.
  const n = d.crops.length, mid = (n - 1) / 2;
  $("#arc").innerHTML = d.crops.map((k, i) => {
    const lift = reduced ? 0 : -(n - 1 - i) * 13;
    const tilt = reduced ? 0 : (i - mid) * 2.2;
    return `<button class="crop ${i === state.sel ? "sel" : ""}" data-i="${i}"
      style="--lift:${lift}px;--tilt:${tilt}deg;
             animation-delay:${i * 30}ms, ${900 + i * 400}ms">
      ${k.requires_irrigation ? '<span class="water">needs water</span>' : ""}
      ${cropArt(k.crop)}
      <span class="rank">${String(k.rank).padStart(2, "0")}</span>
      ${scoreArc(k)}
      <h3>${k.crop}</h3>
      <span class="deva">${k.crop_marathi || "&nbsp;"}</span>
      ${sparkline(k)}
      <span class="band">${band(k)}<span class="yc">${k.yield_class || ""}</span></span>
    </button>`;
  }).join("");

  $("#arc").onclick = e => {
    const b = e.target.closest(".crop"); if (!b) return;
    state.sel = +b.dataset.i;
    $$(".crop").forEach(x => x.classList.toggle("sel", x === b));
    detail(d.crops[state.sel], d);
  };

  detail(d.crops[state.sel], d);

  // sections below the fold fade in as they are reached
  $$("#context .ctx").forEach((el, i) => reveal(el, i * 30));
  $$("#context .num").forEach(n => {
    if (!revealer) return;
    revealer.observe(n);
    setTimeout(() => { if (!n.classList.contains("seen")) countUp(n); }, 1400);
  });
  reveal($(".shortlist-head"));
  reveal($("#rejected"), 60);

  $("#rejected").innerHTML = d.vetoed.length ? `<h4>Vetoed by the agronomic gate</h4>
    <div class="veto">${d.vetoed.map(v =>
      `<span title="${v.reason.replace(/"/g, "&quot;")}">${v.crop}<i>${v.limiting_factor}</i></span>`
    ).join("")}</div>` : "";
}

/* ── detail bento ────────────────────────────────────────────────────── */
const FACTOR_LABEL = { rain: "water", temp: "temp", pH: "pH", depth: "depth",
  drainage: "drain", salinity: "salt", LGP: "season", texture: "texture" };

function detail(k, d) {
  const f = k.factors || {};
  const keys = Object.keys(f);
  const minV = keys.length ? Math.min(...keys.map(x => f[x])) : 1;

  const liebig = keys.length ? `
    <div class="w w-liebig">
      <h4>Liebig's minimum — why this score</h4>
      <div class="liebig">
        ${keys.map(x => `
          <div class="stave ${f[x] === minV ? "min sev-" + severity(minV) : ""}"
               tabindex="0">
            <span class="tip">${factorTip(x, f[x], k)}</span>
            <span class="v">${f[x].toFixed(2)}</span>
            <span class="bar" data-h="${Math.max(f[x] * 100, 2)}"></span>
            <span class="l">${FACTOR_LABEL[x] || x}</span>
          </div>`).join("")}
        <span class="waterline sev-${severity(minV)}" style="bottom:30px"><b>score ${minV.toFixed(2)}</b></span>
      </div>
      <p class="liebig-note">${k.reason}</p>
    </div>` : "";

  const yc = k.yield_class;
  const yieldW = `
    <div class="w w-yield">
      <h4>Yield outlook</h4>
      ${yc ? `<p class="yclass ${yc}">${yc}</p>
        <p class="yield-sub">the norm for ${k.crop} · ${Math.round((k.yield_class_confidence || 0) * 100)}% confidence</p>` : ""}
      ${k.yield_abstained ? `<div class="abstain">${k.yield_interval_note}</div>`
      : k.yield_p50_t_ha != null ? `
        <div class="range">
          <span class="fill" style="left:0;right:0"></span>
          <span class="mid" style="left:${pct(k)}%"></span>
        </div>
        <div class="range-nums"><span>${k.yield_p10_t_ha} t/ha</span>
          <span>median ${k.yield_p50_t_ha}</span><span>${k.yield_p90_t_ha} t/ha</span></div>` : ""}
      <p class="yield-sub" style="margin:18px 0 0">${k.yield_regime || ""}</p>
    </div>`;

  const fert = k.fertiliser;
  let doseW = "", splitW = "";
  if (fert && fert.available) {
    const t = fert.interpolated_target_kg_ha || {};
    const mix = fert.recommended_mix || {};
    doseW = `
      <div class="w w-dose">
        <h4>Fertiliser dose · soil class ${fert.soil_class}
          ${fert.estimated ? '<span class="est">state-median estimate</span>' : ""}</h4>
        ${npkBar(t)}
        ${mix.feasible ? `<div class="mix">${Object.entries(mix.products_kg_ha)
          .map(([p, q]) => `<span>${p} <i>${Math.round(q)} kg</i></span>`).join("")}</div>
          <p class="cost">Least-cost mix · <b>₹${Math.round(mix.cost_inr_per_ha).toLocaleString("en-IN")}</b> per hectare</p>` : ""}
      </div>`;

    const s = fert.nitrogen_schedule;
    if (s) {
      // lay the doses along the season the way a calendar does: the position
      // of each split is the information, and a vertical list hides it
      const days = s.schedule.map(x => {
        const m = /(\d+)\s*-\s*(\d+)\s*DAS/i.exec(x.stage);
        return m ? (+m[1] + +m[2]) / 2 : 0;
      });
      const span = Math.max(...days, 70) * 1.12;
      const maxN = Math.max(...s.schedule.map(x => x.n_kg_ha), 1);
      splitW = `
      <div class="w w-split">
        <h4>Nitrogen schedule · <em class="risk ${s.leach_risk_band}">${
          s.leach_risk_band} leaching risk</em></h4>
        <div class="timeline">
          <span class="track"></span>
          ${s.schedule.map((x, i) => `
            <span class="drop" style="left:${(days[i] / span) * 100}%">
              <b style="height:${20 + (x.n_kg_ha / maxN) * 40}px"></b>
              <u>${x.n_kg_ha}</u><i>${x.stage}</i></span>`).join("")}
        </div>
        <div class="tl-axis"><span>sowing</span><span>${Math.round(span)} days</span></div>
        <p class="leach">${s.rationale}</p>
        <p class="leach basal">Phosphorus and potassium: ${s.phosphorus_potassium}.</p>
      </div>`;
    }
  } else if (fert) {
    doseW = `<div class="w w-dose"><h4>Fertiliser dose</h4>
      <div class="abstain">${fert.note || "No published recommendation for this crop here."}</div></div>`;
  }

  const micro = d.micronutrients.length ? `
    <div class="w w-micro">
      <h4>Micronutrients — the six components the government table ignores</h4>
      <div class="micro">${d.micronutrients.map(m => {
        const C = 2 * Math.PI * 26, off = C * (1 - m.deficient_pct / 100);
        return `
        <div class="mc ${m.priority}">
          <div class="dial">
            <svg viewBox="0 0 64 64" aria-hidden="true">
              <circle class="bg" cx="32" cy="32" r="26"/>
              <circle class="fg" cx="32" cy="32" r="26" transform="rotate(-90 32 32)"
                stroke-dasharray="${C.toFixed(1)}" stroke-dashoffset="${C.toFixed(1)}"
                data-off="${off.toFixed(1)}"/>
            </svg>
            <span class="sym">${m.component}</span>
          </div>
          <div class="mc-body">
            <p class="pct"><b>${m.deficient_pct}%</b> of samples deficient</p>
            <p class="rate">${m.product} · ${m.rate_kg_ha} kg/ha</p>
            <p class="note">${m.note}</p>
            <span class="tag">${m.priority}</span>
          </div>
        </div>`; }).join("")}</div>
    </div>` : "";

  $("#detail").innerHTML = liebig + yieldW + doseW + splitW + micro;
  $$(".w").forEach((w, i) => w.style.animationDelay = `${i * 35}ms`);

  requestAnimationFrame(() => {
    $$(".npk-bar span").forEach(b => b.style.width = b.dataset.w + "%");
    $$(".stave .bar").forEach(b => b.style.height = b.dataset.h + "%");
    $$(".mc .dial .fg").forEach(m => m.style.strokeDashoffset = m.dataset.off);
    const wl = $(".waterline");
    // the staves occupy the box above the 30px label gutter, so the fill line
    // sits at that fraction of the remaining height
    if (wl) wl.style.bottom = `calc(30px + (100% - 30px) * ${minV})`;
  });
}

/* The tightest factor is always highlighted, but colour follows severity —
   marking a 0.76 factor in veto-red when the copy says "not constraining"
   tells the reader two different things at once. */
function severity(v) { return v < 0.5 ? "bad" : v < 0.75 ? "warn" : "ok"; }

/* Each stave states the measurement behind it, not just its score. */
const TIP_EVIDENCE = {
  rain: e => `${Math.round(e.effective_water_mm)} mm available`,
  temp: e => `${e.tmean_c} \u00B0C mean`,
  pH: e => `pH ${e.pH}`,
  depth: e => `${Math.round(e.depth_mm)} mm profile`,
  drainage: e => `drainage class ${e.drainage_ord} of 6`,
  salinity: e => `${e.saline_pct}% saline samples`,
  LGP: e => `${Math.round(e.lgp_days)} day growing period`,
  texture: () => "soil texture preference",
};

function factorTip(factor, value, k) {
  const ev = k.evidence || {};
  const fn = TIP_EVIDENCE[factor];
  const detail = fn && Object.keys(ev).length ? fn(ev) : "";
  const verdict = value >= 0.99 ? "not constraining"
    : value < 0.5 ? "limiting" : "tightening";
  return `<b>${FACTOR_LABEL[factor] || factor} ${value.toFixed(2)}</b> — ${verdict}${
    detail ? `<br>${detail}` : ""}`;
}

/* The card carries the same evidence the detail widget expands: eight staves
   in miniature, so the shortest one is visible before you click. */
function sparkline(k) {
  const f = k.factors || {};
  const keys = Object.keys(f);
  if (!keys.length) return '<span class="spark" aria-hidden="true"></span>';
  return `<span class="spark" aria-hidden="true">${keys.map(x => {
    const v = f[x], cls = v < 0.5 ? "lo" : v < 0.75 ? "mid" : "";
    return `<i class="${cls}" style="height:${Math.max(v * 100, 6)}%"></i>`;
  }).join("")}</span>`;
}

function scoreArc(k) {
  const v = Math.max(0, Math.min(1, k.rule_score ?? 0));
  const C = 2 * Math.PI * 15;
  const tone = v < 0.5 ? "var(--laterite)" : v < 0.75 ? "var(--haldi)" : "var(--kharif)";
  return `<svg class="score-arc" viewBox="0 0 38 38" aria-hidden="true">
    <circle class="bg" cx="19" cy="19" r="15"/>
    <circle class="fg" cx="19" cy="19" r="15" transform="rotate(-90 19 19)"
      stroke="${tone}" stroke-dasharray="${C.toFixed(1)}"
      stroke-dashoffset="${(C * (1 - v)).toFixed(1)}"/>
    <text x="19" y="19">${Math.round(v * 100)}</text></svg>`;
}

/* A crop that only needs water is not unsuitable land. Showing its rainfed
   class "N" beside a "needs water" badge tells the reader two contradictory
   things, so the badge speaks for those and the class is suppressed. */
function band(k) {
  return k.requires_irrigation
    ? `<i class="dot irr"></i>irrigate`
    : `<i class="dot ${k.suitability_class}"></i>${k.suitability_class}`;
}

/* Three numbers in three boxes make you do the ratio in your head. The bar
   is the ratio — and N:P:K balance is what an agronomist reads first. */
function npkBar(t) {
  const parts = [["n", "N", t.N], ["p", "P₂O₅", t.P2O5], ["k", "K₂O", t.K2O]]
    .map(([c, label, v]) => ({ c, label, v: Number(v) || 0 }));
  const total = parts.reduce((a, b) => a + b.v, 0);
  if (!total) return "";
  return `<div class="npk-bar">${parts.map(p =>
      `<span class="${p.c}" data-w="${(p.v / total) * 100}">${
        p.v / total > 0.13 ? Math.round(p.v) : ""}</span>`).join("")}</div>
    <div class="npk-key">${parts.map(p =>
      `<span><b class="${p.c}"></b>${p.label} <u>${
        Math.round(p.v)} kg/ha</u></span>`).join("")}</div>`;
}

function pct(k) {
  const span = k.yield_p90_t_ha - k.yield_p10_t_ha;
  return span > 0 ? Math.min(97, Math.max(3, ((k.yield_p50_t_ha - k.yield_p10_t_ha) / span) * 100)) : 50;
}
