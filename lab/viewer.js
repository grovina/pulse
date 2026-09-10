(() => {
  const W = 1100, H = 860, PAD = 70;
  const state = {
    graph: null,
    run: null,
    frame: 0,
    explode: 0,
    selected: null,
    playing: false,
    lastTs: 0,
    playTimer: 0,
    playhead: 0,
    physAcc: 0,
    busy: false,
    followLive: true,
    asleep: false,
    markerIndex: {},
    gutIndex: {},
    duoIndex: {},
    layers: { coupling: true, carbon: true, bile: true },
  };

  const $ = (id) => document.getElementById(id);

  const svgEl = (name, attrs = {}) => {
    const el = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
    return el;
  };

  const toXY = (m) => {
    const cx = 0.5, cy = 0.48;
    const e = 1 + state.explode * 0.42;
    const x = cx + (m.x - cx) * e;
    const y = cy + (m.y - cy) * e;
    return [PAD + x * (W - 2 * PAD), PAD + y * (H - 2 * PAD)];
  };

  const explodeScale = () => 1 + state.explode * 0.42;

  const waypointPos = (wp) => {
    const mod = state.graph.modules.find((m) => m.id === wp.module);
    const [x, y] = toXY(mod);
    const e = explodeScale();
    return [x + wp.ox * e * (W - 2 * PAD), y + wp.oy * e * (H - 2 * PAD)];
  };

  const clock = (min) => {
    const t = ((min % 1440) + 1440) % 1440;
    const h = Math.floor(t / 60);
    const m = Math.floor(t % 60);
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  };

  const signColor = (sign, alpha) => {
    if (sign < 0) return `rgba(74, 138, 130, ${alpha})`;
    if (sign > 0) return `rgba(194, 74, 58, ${alpha})`;
    return `rgba(201, 162, 39, ${alpha})`;
  };

  const heat = (z) => Math.min(1, Math.abs(z) / 2.2);

  const hexAlpha = (hex, a) => {
    const n = hex.replace("#", "");
    const r = parseInt(n.slice(0, 2), 16);
    const g = parseInt(n.slice(2, 4), 16);
    const b = parseInt(n.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  };

  const channelActivity = (ch, frame) => {
    if (!frame) return 0;
    if (ch in state.gutIndex) {
      const v = frame.gut[state.gutIndex[ch]];
      if (ch.endsWith("nutrient_flag")) return v > 0.5 ? 1 : 0;
      return Math.min(1, Math.abs(v) / 2.5);
    }
    if (ch in state.duoIndex) {
      const v = frame.duo?.[state.duoIndex[ch]] ?? 0;
      return Math.min(1, Math.abs(v) / 0.6);
    }
    const i = state.markerIndex[ch];
    if (i == null) return 0;
    return heat(frame.z[i]);
  };

  const moduleHeat = (mod, frame) => {
    if (!frame) return 0;
    let h = 0;
    for (const id of mod.markers) h = Math.max(h, heat(frame.z[state.markerIndex[id]]));
    for (const id of mod.channels) h = Math.max(h, channelActivity(id, frame));
    return h;
  };

  const satPos = (mod, i, n) => {
    const [x, y] = toXY(mod);
    const a = -Math.PI / 2 + (i / Math.max(1, n)) * Math.PI * 2;
    return [x + Math.cos(a) * 108, y + Math.sin(a) * 108];
  };

  const sourcePos = (edge) => {
    const src = state.graph.modules.find((m) => m.id === edge.source_module);
    if (state.selected === edge.source_module) {
      const ids = [...src.markers, ...src.channels];
      const i = ids.indexOf(edge.source);
      if (i >= 0) return satPos(src, i, ids.length);
    }
    return toXY(src);
  };

  const quad = (x1, y1, x2, y2, fan) => {
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy) || 1;
    const cx = mx - (dy / len) * fan;
    const cy = my + (dx / len) * fan;
    return `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;
  };

  function drawLoops(frame) {
    const loopsG = $("loops");
    const wayG = $("waypoints");
    loopsG.replaceChildren();
    wayG.replaceChildren();
    const loops = state.graph.loops || [];
    for (const loop of loops) {
      if (!state.layers[loop.id]) continue;
      const pack = frame?.flux?.[loop.id] || {};
      const byId = Object.fromEntries(loop.waypoints.map((w) => [w.id, w]));

      loop.flows.forEach((flow, fi) => {
        const a = byId[flow.from];
        const b = byId[flow.to];
        if (!a || !b) return;
        const [x1, y1] = waypointPos(a);
        const [x2, y2] = waypointPos(b);
        const v = Math.abs(pack[flow.id] ?? 0);
        const act = Math.min(1, v / loop.scale);
        const leak = Math.abs(pack.residual ?? 0) > loop.residual_warn;
        if (act < 0.02 && a.kind !== "hole" && b.kind !== "hole" && a.kind !== "inlet") {
          if (act < 0.004) return;
        }
        const path = svgEl("path", {
          d: quad(x1, y1, x2, y2, ((fi % 5) - 2) * 18),
          fill: "none",
          stroke: leak ? "#c24a3a" : hexAlpha(loop.color, 0.18 + act * 0.72),
          "stroke-width": String(1.1 + act * 5.2),
          "stroke-linecap": "round",
          class: leak ? "flow broken" : (act > 0.06 ? "flow" : ""),
        });
        if (!leak && act > 0.06) {
          path.style.setProperty("--flow-ms", `${Math.round(1400 - act * 850)}ms`);
        }
        path.setAttribute("aria-hidden", "true");
        loopsG.appendChild(path);
      });

      for (const wp of loop.waypoints) {
        const [x, y] = waypointPos(wp);
        if (wp.kind === "anchor") continue;
        if (wp.kind === "hole") {
          const arriving = loop.flows
            .filter((f) => f.to === wp.id)
            .reduce((s, f) => s + Math.abs(pack[f.id] ?? 0), 0);
          const act = Math.min(1, arriving / loop.scale);
          wayG.appendChild(svgEl("circle", {
            cx: x, cy: y, r: 16,
            class: "hole-ring",
            stroke: hexAlpha(loop.color, 0.35 + act * 0.55),
            "stroke-width": "1.4",
          }));
          wayG.appendChild(svgEl("circle", {
            cx: x, cy: y, r: 7,
            fill: "none",
            stroke: hexAlpha(loop.color, 0.5 + act * 0.5),
            "stroke-width": "1.6",
          }));
        } else if (wp.kind === "inlet") {
          const leaving = loop.flows
            .filter((f) => f.from === wp.id)
            .reduce((s, f) => s + Math.abs(pack[f.id] ?? 0), 0);
          const act = Math.min(1, leaving / loop.scale);
          wayG.appendChild(svgEl("circle", {
            cx: x, cy: y, r: 5,
            fill: hexAlpha(loop.color, 0.25 + act * 0.55),
            stroke: loop.color,
            "stroke-width": "1.2",
          }));
          wayG.appendChild(svgEl("path", {
            d: `M ${x - 6} ${y} L ${x + 6} ${y} M ${x} ${y - 6} L ${x} ${y + 6}`,
            stroke: loop.color,
            "stroke-width": "1.4",
            fill: "none",
          }));
        } else {
          wayG.appendChild(svgEl("circle", {
            cx: x, cy: y, r: 5.5,
            fill: hexAlpha(loop.color, 0.55),
            stroke: loop.color,
            "stroke-width": "1.1",
          }));
        }
        const skipLabel = Boolean(
          wp.marker
          && state.selected
          && state.layers.coupling
          && state.graph.modules.find((m) => m.id === state.selected)?.markers.includes(wp.marker)
        );
        if (!skipLabel) {
          const label = svgEl("text", {
            x: x + 12, y: y + 4,
            fill: "#8f877c",
            "font-size": "10",
            "font-family": "Public Sans, sans-serif",
          });
          label.textContent = wp.label;
          wayG.appendChild(label);
        }
      }
    }
  }

  function drawGraph() {
    const g = state.graph;
    const frame = state.run?.frames[state.frame];
    const edgesG = $("edges");
    const nodesG = $("nodes");
    const satsG = $("sats");
    edgesG.replaceChildren();
    nodesG.replaceChildren();
    satsG.replaceChildren();

    if (state.layers.coupling) {
      g.edges.forEach((edge, ei) => {
        const [x1, y1] = sourcePos(edge);
        const tgt = g.modules.find((m) => m.id === edge.target_module);
        const [x2, y2] = toXY(tgt);
        const act = channelActivity(edge.source, frame);
        const path = svgEl("path", {
          d: quad(x1, y1, x2, y2, ((ei % 5) - 2) * 14),
          fill: "none",
          stroke: signColor(edge.sign, 0.22 + act * 0.7),
          "stroke-width": String(1.2 + act * 3.4),
          "stroke-linecap": "round",
        });
        if (state.selected && edge.source_module !== state.selected && edge.target_module !== state.selected) {
          path.setAttribute("opacity", "0.18");
        }
        edgesG.appendChild(path);
      });
    }

    drawLoops(frame);

    for (const mod of g.modules) {
      const [x, y] = toXY(mod);
      const h = moduleHeat(mod, frame);
      const selected = state.selected === mod.id;
      const wrap = svgEl("g", {
        class: "node-hit",
        tabindex: "0",
        role: "button",
        "aria-pressed": selected ? "true" : "false",
        "aria-label": mod.name,
      });
      wrap.addEventListener("click", () => {
        state.selected = state.selected === mod.id ? null : mod.id;
        drawGraph();
        drawReadout();
      });
      wrap.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          wrap.dispatchEvent(new Event("click"));
        }
      });
      wrap.appendChild(svgEl("circle", {
        cx: x, cy: y,
        r: selected ? 40 : 34,
        fill: `rgba(194, 74, 58, ${0.08 + h * 0.45})`,
        stroke: selected ? "#e8dfd0" : `rgba(232, 223, 208, ${0.35 + h * 0.5})`,
        "stroke-width": selected ? "2.2" : "1.2",
      }));
      const t = svgEl("text", {
        x, y: y + 4,
        "text-anchor": "middle",
        fill: "#e8dfd0",
        "font-size": "13",
        "font-family": "Newsreader, Georgia, serif",
      });
      t.textContent = mod.name;
      wrap.appendChild(t);
      nodesG.appendChild(wrap);
    }

    if (state.selected && state.layers.coupling) {
      const mod = g.modules.find((m) => m.id === state.selected);
      const ids = [...mod.markers, ...mod.channels];
      ids.forEach((id, i) => {
        const [x, y] = satPos(mod, i, ids.length);
        const marker = g.markers.find((m) => m.id === id);
        const ch = g.channels.find((c) => c.id === id);
        const label = marker ? marker.label : ch.label;
        const act = channelActivity(id, frame);
        const [mx] = toXY(mod);
        satsG.appendChild(svgEl("circle", {
          cx: x, cy: y, r: 6,
          fill: marker ? marker.color : "#c9a227",
          opacity: String(0.35 + act * 0.65),
        }));
        const txt = svgEl("text", {
          x: x + (x >= mx ? 10 : -10),
          y: y + 3,
          "text-anchor": x >= mx ? "start" : "end",
          fill: "#8f877c",
          "font-size": "10",
          "font-family": "Public Sans, sans-serif",
        });
        txt.textContent = label;
        satsG.appendChild(txt);
      });
    }
  }

  function sparkline(values, at) {
    const w = 120, h = 22, pad = 1.5;
    const svg = svgEl("svg", {
      class: "spark",
      viewBox: `0 0 ${w} ${h}`,
      "aria-hidden": "true",
    });
    const n = values.length;
    if (!n) return svg;
    let lo = values[0], hi = values[0];
    for (const v of values) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const span = hi - lo || 1;
    const xOf = (i) => pad + (i / Math.max(1, n - 1)) * (w - 2 * pad);
    const yOf = (v) => h - pad - ((v - lo) / span) * (h - 2 * pad);
    const pts = (from, to) => {
      const out = [];
      for (let i = from; i <= to; i++) out.push(`${xOf(i).toFixed(2)},${yOf(values[i]).toFixed(2)}`);
      return out.join(" ");
    };
    const iNow = Math.max(0, Math.min(n - 1, at));
    if (n > 1) {
      svg.appendChild(svgEl("polyline", {
        class: "past",
        points: pts(0, n - 1),
        fill: "none",
      }));
    }
    const x = xOf(iNow);
    const y = yOf(values[iNow]);
    svg.appendChild(svgEl("line", { class: "now", x1: x, x2: x, y1: 0, y2: h }));
    svg.appendChild(svgEl("circle", { class: "tip", cx: x, cy: y, r: 1.8 }));
    return svg;
  }

  function seriesRow(name, value, unit, values) {
    const row = document.createElement("div");
    row.className = "row";
    const nm = document.createElement("span");
    nm.className = "name";
    nm.textContent = name;
    const val = document.createElement("span");
    val.className = "val";
    val.textContent = unit ? `${value} ${unit}` : value;
    row.append(nm, sparkline(values, state.frame), val);
    return row;
  }

  function fmt(v) {
    const a = Math.abs(v);
    const d = a >= 100 ? 0 : a >= 10 ? 1 : 2;
    return v.toFixed(d);
  }

  function fmtFlux(v) {
    const a = Math.abs(v);
    if (a >= 1) return v.toFixed(2);
    if (a >= 0.01) return v.toFixed(3);
    if (a === 0) return "0";
    return v.toExponential(1);
  }

  function feelingRow(axis, feel) {
    const el = document.createElement("div");
    const label = feel?.label || "";
    el.className = `feeling is-${label.toLowerCase()}`;
    const word = document.createElement("span");
    word.className = "word";
    word.textContent = label;
    const meta = document.createElement("span");
    meta.className = "meta";
    if (axis === "sleep" && feel?.kind === "derived") {
      meta.textContent = "from the clock";
    } else if (feel?.derived) {
      meta.textContent = "derived";
    } else if (axis === "sleep") {
      meta.textContent = "you set this";
    } else {
      meta.textContent = axis;
    }
    el.append(word, meta);
    return el;
  }

  function drawFeelings(frame) {
    const box = $("feelings");
    box.replaceChildren();
    const f = frame.feelings;
    if (!f) return;
    box.append(
      feelingRow("hunger", f.hunger),
      feelingRow("heat", f.heat),
      feelingRow("tired", f.tired),
      feelingRow("sleep", f.sleep),
    );
  }

  function drawReadout() {
    const run = state.run;
    if (!run) return;
    const frame = run.frames[state.frame];
    $("clock").textContent = clock(frame.clock_min);
    const eaten = (run.meals || []).some((m) => m.t <= frame.t);
    const hours = frame.feelings?.hours_since_meal;
    let since = "";
    if (!eaten) since = "No meal yet.";
    else if (hours != null && hours < 0.2) since = "Just ate.";
    else if (hours != null) since = `${fmt(hours)} h since the last meal.`;
    const f = frame.feelings;
    const words = f
      ? [f.hunger?.label, f.heat?.label, f.tired?.label, f.sleep?.label].filter(Boolean).join(", ")
      : "";
    const looking = !state.followLive && state.frame < run.frames.length - 1;
    $("run-blurb").textContent = looking
      ? `Looking back. Play continues from now. ${since}`.trim()
      : `${words}${since ? `. ${since}` : ""}`.trim();
    const kicker = document.querySelector(".kicker");
    if (kicker && state.model) {
      kicker.textContent = state.model.trained
        ? `Pulse lab · ${state.model.version}`
        : "Pulse lab · untrained student";
    }
    const phase = (run.phases || []).find((p) => frame.t >= p.start_min && frame.t < p.end_min);
    $("phase").textContent = phase ? phase.label : "";
    drawFeelings(frame);

    if (state.busy) {
      $("hint").textContent = "The body is catching up.";
    } else if (state.asleep) {
      $("hint").textContent = "Asleep. Wake to eat or walk.";
    } else if (frame.appearing) {
      $("hint").textContent = "The meal is appearing. Play to live the next minutes.";
    } else {
      $("hint").textContent = "Eat, walk, or lie down. Play steps the model. Nothing past now is shown.";
    }

    const badges = [];
    if (frame.sleep_wake < 0.5) badges.push(["Asleep", "on"]);
    if (frame.appearing) badges.push(["Meal appearing", "on"]);
    if (frame.activity > 0.2) badges.push(["Walking", "on"]);
    const lastMeal = (run.meals || []).filter((m) => m.t <= frame.t).at(-1);
    if (!lastMeal || frame.t - lastMeal.t >= 480) badges.push(["Fasting", "on"]);
    if (frame.clamped.length) {
      const id = frame.clamped[0];
      const m = state.graph.markers.find((x) => x.id === id);
      badges.push([`Clamp ${m?.label ?? id}`, "clamp"]);
    }
    for (const loop of state.graph.loops || []) {
      if (!state.layers[loop.id]) continue;
      const res = frame.flux?.[loop.id]?.residual ?? 0;
      if (Math.abs(res) > loop.residual_warn) badges.push([`${loop.name} leak`, "leak"]);
    }
    $("badges").replaceChildren(...badges.map(([text, cls]) => {
      const el = document.createElement("span");
      el.className = `badge ${cls}`;
      el.textContent = text;
      return el;
    }));

    const rows = $("readout-rows");
    rows.replaceChildren();
    for (const id of state.graph.readout) {
      const i = state.markerIndex[id];
      const m = state.graph.markers[i];
      const series = run.frames.map((fr) => fr.state[i]);
      rows.appendChild(seriesRow(m.label, fmt(frame.state[i]), m.unit, series));
    }

    const ledger = $("ledger");
    ledger.replaceChildren();
    for (const loop of state.graph.loops || []) {
      if (!state.layers[loop.id]) continue;
      const pack = frame.flux?.[loop.id];
      if (!pack) continue;
      const h = document.createElement("h3");
      h.textContent = `${loop.name} ledger`;
      ledger.appendChild(h);
      for (const row of loop.ledger) {
        const series = run.frames.map((fr) => fr.flux?.[loop.id]?.[row.id] ?? 0);
        ledger.appendChild(seriesRow(row.name, fmtFlux(pack[row.id] ?? 0), loop.unit, series));
      }
    }

    const heading = $("module-heading");
    const mrows = $("module-rows");
    mrows.replaceChildren();
    if (!state.selected) {
      heading.textContent = "Select a module";
      return;
    }
    const mod = state.graph.modules.find((m) => m.id === state.selected);
    heading.textContent = mod.name;
    for (const id of mod.markers) {
      const i = state.markerIndex[id];
      const m = state.graph.markers.find((x) => x.id === id);
      const series = run.frames.map((fr) => fr.state[i]);
      mrows.appendChild(seriesRow(m.label, fmt(frame.state[i]), m.unit, series));
    }
    for (const id of mod.channels) {
      const ch = state.graph.channels.find((c) => c.id === id);
      let series;
      let v = 0;
      if (id in state.gutIndex) {
        const gi = state.gutIndex[id];
        series = run.frames.map((fr) => fr.gut[gi]);
        v = frame.gut[gi];
      } else if (id in state.duoIndex) {
        const di = state.duoIndex[id];
        series = run.frames.map((fr) => fr.duo?.[di] ?? 0);
        v = frame.duo?.[di] ?? 0;
      } else {
        continue;
      }
      mrows.appendChild(seriesRow(ch.label, fmtFlux(v), ch.unit, series));
    }
  }

  function setFrame(i, fromPlayhead = false) {
    if (!state.run) return;
    const last = state.run.frames.length - 1;
    state.frame = Math.max(0, Math.min(last, i));
    if (!fromPlayhead) {
      state.playhead = state.frame;
      state.followLive = state.frame >= last;
    }
    $("scrub").value = String(state.frame);
    const frame = state.run.frames[state.frame];
    const live = state.run.frames[state.run.frames.length - 1];
    state.asleep = live.sleep_wake < 0.5;
    syncRestButton();
    drawGraph();
    drawReadout();
  }

  function tick() {
    if (!state.playing) return;
    const ts = performance.now();
    const dt = (ts - state.lastTs) / 1000;
    state.lastTs = ts;
    const speed = Number($("speed").value);
    state.physAcc += speed * Math.min(dt, 0.25);
    const step = state.run?.sample_every_min || 5;
    const horizon = (state.run?.duration_min ?? 1) - 1;
    if (!state.busy && state.physAcc >= step) {
      const minutes = Math.min(Math.floor(state.physAcc), step);
      state.physAcc -= minutes;
      advanceBy(minutes).then((snap) => {
        if (!state.playing) return;
        if (!snap || snap.t >= horizon) setPlaying(false);
      });
    }
  }

  function setPlaying(on) {
    state.playing = on;
    $("play").textContent = on ? "Pause" : "Play";
    $("play").setAttribute("aria-pressed", on ? "true" : "false");
    if (state.playTimer) {
      clearInterval(state.playTimer);
      state.playTimer = 0;
    }
    if (on) {
      state.followLive = true;
      state.physAcc = 0;
      state.lastTs = performance.now();
      state.playTimer = setInterval(tick, 50);
      tick();
    }
  }

  async function advanceBy(minutes) {
    const snap = await post("/api/advance", { minutes });
    if (snap) applySnapshot(snap, true);
    return snap;
  }

  function applySnapshot(snap, followLive = true) {
    const run = snap.run;
    state.run = run;
    state.model = snap.model;
    state.asleep = snap.asleep;
    state.followLive = followLive;
    state.gutIndex = Object.fromEntries(run.gut_channel_ids.map((id, i) => [id, i]));
    state.duoIndex = Object.fromEntries((run.duodenal_channel_ids || []).map((id, i) => [id, i]));
    const last = run.frames.length - 1;
    state.frame = last;
    state.playhead = last;
    $("scrub").max = String(last);
    $("scrub").value = String(state.frame);
    for (const btn of $("protocols").querySelectorAll("button")) {
      btn.setAttribute("aria-pressed", btn.dataset.id === snap.protocol_id ? "true" : "false");
    }
    syncRestButton();
    drawGraph();
    drawReadout();
  }

  function syncRestButton() {
    const rest = $("rest");
    rest.textContent = state.asleep ? "Wake" : "Lie down";
    $("eat").disabled = state.asleep;
    $("walk").disabled = state.asleep;
  }

  async function post(path, extra = {}) {
    if (state.busy) return null;
    state.busy = true;
    if (state.run) drawReadout();
    try {
      const res = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(extra),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = body.detail;
        $("hint").textContent = typeof detail === "string" ? detail : "The body did not take that.";
        return null;
      }
      return body;
    } finally {
      state.busy = false;
    }
  }

  async function resetProtocol(id) {
    setPlaying(false);
    const snap = await post("/api/reset", { protocol: id });
    if (snap) applySnapshot(snap, true);
  }

  function bindLayers() {
    const nav = $("layers");
    nav.replaceChildren();
    const defs = [
      { id: "coupling", name: "Coupling" },
      ...(state.graph.loops || []).map((l) => ({ id: l.id, name: l.name })),
    ];
    for (const def of defs) {
      if (!(def.id in state.layers)) state.layers[def.id] = true;
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.layer = def.id;
      b.textContent = def.name;
      b.setAttribute("aria-pressed", state.layers[def.id] ? "true" : "false");
      b.addEventListener("click", () => {
        state.layers[def.id] = !state.layers[def.id];
        b.setAttribute("aria-pressed", state.layers[def.id] ? "true" : "false");
        drawGraph();
        drawReadout();
      });
      nav.appendChild(b);
    }
  }

  function bind() {
    $("play").addEventListener("click", () => setPlaying(!state.playing));
    $("scrub").addEventListener("input", (e) => {
      setPlaying(false);
      setFrame(Number(e.target.value));
    });
    $("explode").addEventListener("input", (e) => {
      state.explode = Number(e.target.value) / 100;
      drawGraph();
    });
    $("eat").addEventListener("click", async () => {
      const snap = await post("/api/eat", { plate: $("plate").value });
      if (snap) applySnapshot(snap, true);
    });
    $("walk").addEventListener("click", async () => {
      const snap = await post("/api/walk", { minutes: 30, intensity: 0.45 });
      if (snap) applySnapshot(snap, true);
    });
    $("rest").addEventListener("click", async () => {
      const snap = await post(state.asleep ? "/api/wake" : "/api/rest");
      if (snap) applySnapshot(snap, true);
    });
    window.addEventListener("keydown", (e) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.key === " ") {
        e.preventDefault();
        setPlaying(!state.playing);
      }
      if (e.key === "e" || e.key === "E") {
        e.preventDefault();
        $("eat").click();
      }
      if (e.key === "w" || e.key === "W") {
        e.preventDefault();
        $("walk").click();
      }
      if (e.key === "l" || e.key === "L") {
        e.preventDefault();
        $("rest").click();
      }
      if (e.key === "ArrowRight") {
        setPlaying(false);
        setFrame(state.frame + 1);
      }
      if (e.key === "ArrowLeft") {
        setPlaying(false);
        setFrame(state.frame - 1);
      }
    });
  }

  async function main() {
    bind();
    const graphRes = await fetch("/api/graph");
    if (!graphRes.ok) {
      $("run-blurb").textContent = "Start the lab with uv run python scripts/lab_viewer.py";
      return;
    }
    state.graph = await graphRes.json();
    state.graph.markers.forEach((m, i) => { state.markerIndex[m.id] = i; });
    bindLayers();
    const protocols = await (await fetch("/api/protocols")).json();
    const nav = $("protocols");
    for (const p of protocols) {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.id = p.id;
      b.textContent = p.title;
      b.addEventListener("click", () => resetProtocol(p.id));
      nav.appendChild(b);
    }
    const snap = await (await fetch("/api/snapshot")).json();
    applySnapshot(snap, true);
  }

  main().catch((err) => {
    $("run-blurb").textContent = `Could not start the lab: ${err.message}`;
  });
})();
