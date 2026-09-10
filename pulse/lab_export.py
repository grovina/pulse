"""Lab-viewer graph and teacher runs. Schema follows `types.py`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .knowledge.full_body import (
    PatientParams,
    bile_fluxes,
    duodenal_delivery,
    glucose_fluxes,
    resolve_derived_params,
    simulate_full_body,
)
from .knowledge.textbook_scenarios.flow_story_protocol import (
    DIETARY_CARB_FLOW_DURATION_MIN,
    DIETARY_CARB_FLOW_MEALS,
    DIETARY_CARB_FLOW_START_HOUR,
    dietary_carb_flow_phases_for_ui,
)
from .types import (
    COUPLING_GRAPH,
    DUODENAL_CHANNEL_IDS,
    GUT_CHANNEL_IDS,
    MARKER_INDEX,
    MARKERS,
    MODULE_COUPLING_CHANNELS,
    MODULE_MARKER_IDS,
    NORM_CENTER,
    NORM_SCALE,
    PHYSIOLOGICAL_MAX,
    PHYSIOLOGICAL_MIN,
    System,
)

_EDGE_SIGN = {(e.source_marker, e.target_module): e.sign_prior for e in COUPLING_GRAPH}

LAB_DIR = Path(__file__).resolve().parent.parent / "lab"
SAMPLE_EVERY_MIN = 5

# Body-ish layout in [0, 1]: head at the top, viscera in the middle, vitals
# in the chest. These coordinates are the attachment points a later anatomical
# layer would parent to meshes.
MODULE_LAYOUT: dict[str, tuple[float, float]] = {
    "stress": (0.50, 0.16),
    "cardiovascular": (0.34, 0.34),
    "respiratory": (0.66, 0.34),
    "appetite": (0.22, 0.52),
    "metabolic": (0.50, 0.52),
    "hepatobiliary": (0.78, 0.52),
    "gut": (0.50, 0.70),
    "thermoreg": (0.22, 0.78),
}

MODULE_TITLE: dict[str, str] = {
    "gut": "Gut",
    "metabolic": "Metabolic",
    "appetite": "Appetite",
    "stress": "Stress",
    "cardiovascular": "Heart",
    "thermoreg": "Heat",
    "respiratory": "Breath",
    "hepatobiliary": "Bile",
}

MODULE_KIND: dict[str, str] = {
    "gut": "kernel",
    "metabolic": "mass_action",
    "appetite": "mass_action",
    "stress": "mass_action",
    "cardiovascular": "learned",
    "thermoreg": "learned",
    "respiratory": "learned",
    "hepatobiliary": "mass_action",
}

CHANNEL_META: dict[str, dict[str, str]] = {
    "gut.glucose_appearance": {
        "name": "Glucose appearance", "unit": "mg/dL/min",
    },
    "gut.lipid_appearance": {"name": "Lipid appearance", "unit": "g/min"},
    "gut.amino_appearance": {"name": "Amino appearance", "unit": "g/min"},
    "gut.nutrient_flag": {"name": "Nutrient flag", "unit": ""},
    "duodenal.fat": {"name": "Duodenal fat", "unit": "g/min"},
    "duodenal.protein": {"name": "Duodenal protein", "unit": "g/min"},
    "duodenal.carb": {"name": "Duodenal carbohydrate", "unit": "g/min"},
}

# Graph / satellite labels. MarkerDef.name is the readout name; these are
# the same words with list-disambiguators stripped so they fit a node.
GRAPH_LABEL: dict[str, str] = {
    "glucose": "Glucose",
    "ffa": "FFA",
    "bhb": "BHB",
    "hepatic_output": "Hepatic output",
    "hrv": "HRV",
    "temp": "Temperature",
    "insulin_action": "Insulin action",
    "insulin_slow": "Slow insulin",
    "cck": "CCK",
    "gallbladder_bile": "Gallbladder",
    "intestinal_bile": "Intestine",
    "bile_acids": "Serum BA",
    "mitochondrial_capacity": "Mitochondria",
}

GUT_SERIES_INDEX = {
    "gut.glucose_appearance": 0,
    "gut.lipid_appearance": 1,
    "gut.amino_appearance": 2,
    "gut.nutrient_flag": 3,
}

# Conserved loops drawn on the coupling graph. Waypoint (ox, oy) is in the
# same [0, 1] space as MODULE_LAYOUT, relative to the parent module, so
# explode moves them with the body. `anchor` sits on the module; `pool` is
# a named satellite; `hole` is an intended drain; `inlet` is de-novo mass.
LOOPS: list[dict[str, Any]] = [
    {
        "id": "carbon",
        "name": "Carbon",
        "unit": "g/min",
        "color": "#c9a227",
        "scale": 0.35,
        "residual_warn": 0.02,
        "waypoints": [
            {"id": "gut", "kind": "anchor", "module": "gut", "ox": 0.0, "oy": 0.0,
             "label": "Gut"},
            {"id": "glucose", "kind": "pool", "module": "metabolic", "marker": "glucose",
             "ox": 0.0, "oy": -0.08},
            {"id": "liver_glycogen", "kind": "pool", "module": "metabolic",
             "marker": "liver_glycogen", "ox": 0.11, "oy": 0.01},
            {"id": "muscle_glycogen", "kind": "pool", "module": "metabolic",
             "marker": "muscle_glycogen", "ox": -0.12, "oy": 0.04},
            {"id": "gng", "kind": "inlet", "module": "hepatobiliary",
             "ox": -0.09, "oy": -0.07, "label": "Gluconeogenesis"},
            {"id": "oxidized", "kind": "hole", "module": "thermoreg",
             "ox": 0.08, "oy": 0.0, "label": "Oxidized"},
        ],
        "flows": [
            {"id": "ra", "from": "gut", "to": "glucose", "label": "appearance"},
            {"id": "syn_L", "from": "glucose", "to": "liver_glycogen", "label": "to liver"},
            {"id": "glyco", "from": "liver_glycogen", "to": "glucose", "label": "glycogenolysis"},
            {"id": "gng_rel", "from": "gng", "to": "glucose", "label": "GNG released"},
            {"id": "gng_divert", "from": "gng", "to": "liver_glycogen", "label": "GNG to glycogen"},
            {"id": "syn_M", "from": "glucose", "to": "muscle_glycogen", "label": "to muscle"},
            {"id": "uptake", "from": "glucose", "to": "oxidized", "label": "oxidized"},
            {"id": "brk_M", "from": "muscle_glycogen", "to": "oxidized", "label": "muscle oxidation"},
        ],
        "ledger": [
            {"id": "ra", "name": "Appearance"},
            {"id": "oxidized", "name": "Oxidized"},
            {"id": "residual", "name": "Residual"},
        ],
    },
    {
        "id": "bile",
        "name": "Bile",
        "unit": "mmol/min",
        "color": "#8aa14a",
        "scale": 0.06,
        "residual_warn": 1e-4,
        "waypoints": [
            {"id": "gallbladder", "kind": "pool", "module": "hepatobiliary",
             "marker": "gallbladder_bile", "ox": 0.09, "oy": -0.03},
            {"id": "intestine", "kind": "pool", "module": "gut",
             "marker": "intestinal_bile", "ox": 0.13, "oy": 0.0},
            {"id": "liver", "kind": "anchor", "module": "hepatobiliary",
             "ox": 0.0, "oy": 0.0, "label": "Liver"},
            {"id": "serum", "kind": "pool", "module": "hepatobiliary",
             "marker": "bile_acids", "ox": 0.03, "oy": -0.13},
            {"id": "synth", "kind": "inlet", "module": "hepatobiliary",
             "ox": -0.10, "oy": 0.06, "label": "Synthesis"},
            {"id": "faecal", "kind": "hole", "module": "gut",
             "ox": 0.16, "oy": 0.10, "label": "Faecal"},
        ],
        "flows": [
            {"id": "empty", "from": "gallbladder", "to": "intestine", "label": "emptying"},
            {"id": "ileal", "from": "intestine", "to": "liver", "label": "ileal return"},
            {"id": "fill", "from": "liver", "to": "gallbladder", "label": "filling"},
            {"id": "direct", "from": "liver", "to": "intestine", "label": "direct bile"},
            {"id": "spill", "from": "liver", "to": "serum", "label": "spillover"},
            {"id": "serum_return", "from": "serum", "to": "liver", "label": "serum return"},
            {"id": "faecal", "from": "intestine", "to": "faecal", "label": "faecal loss"},
            {"id": "synth", "from": "synth", "to": "liver", "label": "synthesis"},
        ],
        "ledger": [
            {"id": "synth", "name": "Synthesis"},
            {"id": "faecal", "name": "Faecal loss"},
            {"id": "residual", "name": "Residual"},
        ],
    },
]


def _r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def _marker_label(marker_id: str) -> str:
    if marker_id in GRAPH_LABEL:
        return GRAPH_LABEL[marker_id]
    return next(m.name for m in MARKERS if m.id == marker_id)


def _loops() -> list[dict[str, Any]]:
    """Copy loop specs and fill pool labels from the marker display name."""
    out = []
    for loop in LOOPS:
        loop = dict(loop)
        waypoints = []
        for wp in loop["waypoints"]:
            wp = dict(wp)
            if "marker" in wp:
                wp["label"] = _marker_label(wp["marker"])
            waypoints.append(wp)
        loop["waypoints"] = waypoints
        out.append(loop)
    return out


def _channel_module(channel: str) -> str:
    if channel.startswith("gut.") or channel.startswith("duodenal."):
        return "gut"
    for system, ids in MODULE_MARKER_IDS.items():
        if channel in ids:
            return system
    raise KeyError(f"channel {channel!r} is not a marker or gut/duodenal port")


def _edge_sign(channel: str, target: str) -> int:
    if (channel, target) in _EDGE_SIGN:
        return int(_EDGE_SIGN[(channel, target)])
    if channel.startswith("gut.") or channel.startswith("duodenal."):
        return 1
    return 0


def build_graph() -> dict[str, Any]:
    """Modules, markers, channels, edges — the coupling layout `forward()` reads."""
    modules = []
    module_ids = ["gut", *[s.value for s in System]]
    for mid in module_ids:
        modules.append({
            "id": mid,
            "name": MODULE_TITLE[mid],
            "kind": MODULE_KIND[mid],
            "x": MODULE_LAYOUT[mid][0],
            "y": MODULE_LAYOUT[mid][1],
            "markers": list(MODULE_MARKER_IDS.get(mid, [])),
            "channels": [
                ch for ch, owner in (
                    (c, _channel_module(c))
                    for c in (*GUT_CHANNEL_IDS, *DUODENAL_CHANNEL_IDS)
                )
                if owner == mid
            ],
        })

    markers = [
        {
            "id": m.id,
            "name": m.name,
            "label": _marker_label(m.id),
            "unit": m.unit,
            "module": m.system.value,
            "typical": m.typical,
            "ui_system": m.ui_system,
            "min": m.min,
            "max": m.max,
            "color": m.color,
            "phys_min": float(PHYSIOLOGICAL_MIN[i]),
            "phys_max": float(PHYSIOLOGICAL_MAX[i]),
            "norm_center": float(NORM_CENTER[i]),
            "norm_scale": float(NORM_SCALE[i]),
        }
        for i, m in enumerate(MARKERS)
    ]

    channels = [
        {
            "id": ch,
            "name": CHANNEL_META[ch]["name"],
            "label": CHANNEL_META[ch]["name"],
            "unit": CHANNEL_META[ch]["unit"],
            "module": _channel_module(ch),
        }
        for ch in (*GUT_CHANNEL_IDS, *DUODENAL_CHANNEL_IDS)
    ]

    edges = []
    for target, chans in MODULE_COUPLING_CHANNELS.items():
        for ch in chans:
            edges.append({
                "source": ch,
                "source_module": _channel_module(ch),
                "target_module": target,
                "sign": _edge_sign(ch, target),
            })

    return {
        "schema": "pulse.lab.graph.v2",
        "modules": modules,
        "markers": markers,
        "channels": channels,
        "edges": edges,
        "loops": _loops(),
        "readout": ["glucose", "hr", "ghrelin", "cortisol", "temp"],
    }


def _sleep_mask(
    duration_min: int, start_hour: float, bed_hour: float = 23.0, wake_hour: float = 7.0,
) -> np.ndarray:
    sw = np.ones(duration_min, dtype=np.float32)
    for t in range(duration_min):
        hour = (start_hour + t / 60.0) % 24.0
        if bed_hour > wake_hour:
            asleep = hour >= bed_hour or hour < wake_hour
        else:
            asleep = bed_hour <= hour < wake_hour
        if asleep:
            sw[t] = 0.0
    kernel = np.ones(20, dtype=np.float32) / 20.0
    return np.clip(np.convolve(sw, kernel, mode="same"), 0.0, 1.0).astype(np.float32)


def _clock_minutes(start_hour: float, t: int) -> float:
    return (start_hour * 60.0 + t) % 1440.0


def _protocol_specs() -> dict[str, dict[str, Any]]:
    day_meals = [
        (120.0, 50.0, 12.0, 18.0),
        (420.0, 70.0, 20.0, 25.0),
        (780.0, 80.0, 25.0, 30.0),
    ]
    return {
        "day": {
            "title": "Eucaloric day",
            "blurb": "Three meals, overnight sleep, a walk at 17:00.",
            "duration_min": 1440,
            "start_hour": 6.0,
            "meals": day_meals,
            "walk": (660, 30, 0.45),
        },
        "fast": {
            "title": "24 h fast",
            "blurb": "No meals. Sleep as usual. Fat mass and ketones should move.",
            "duration_min": 1440,
            "start_hour": 6.0,
            "meals": [],
            "walk": None,
        },
        "dawn": {
            "title": "Pre-dawn window",
            "blurb": "03:00–09:00 asleep until 07:00. Cortisol nadir, then rise.",
            "duration_min": 360,
            "start_hour": 3.0,
            "meals": [],
            "walk": None,
        },
        "meal": {
            "title": "Carbohydrate meal",
            "blurb": "The dietary-carb flow story: 50 g at t+30 from 08:00.",
            "duration_min": DIETARY_CARB_FLOW_DURATION_MIN,
            "start_hour": DIETARY_CARB_FLOW_START_HOUR,
            "meals": list(DIETARY_CARB_FLOW_MEALS),
            "walk": None,
        },
    }


def _bergman_X(params: PatientParams, state: np.ndarray) -> float:
    """Recover Bergman's X from the stored insulin_action marker."""
    stored = float(state[MARKER_INDEX["insulin_action"]])
    return stored * params.Si * 10.0 if params.Si != 0.0 else 0.0


def _carbon_flux(params: PatientParams, state: np.ndarray, ra: float, act: float) -> dict[str, float]:
    """Carbon weather in g/min. Residual is the ledger identity, ~0 on the teacher."""
    mi = MARKER_INDEX
    fl = glucose_fluxes(
        params,
        float(state[mi["glucose"]]),
        float(state[mi["insulin"]]),
        _bergman_X(params, state),
        float(state[mi["glucagon"]]),
        float(state[mi["cortisol"]]),
        float(state[mi["ffa"]]),
        float(state[mi["liver_glycogen"]]),
        float(state[mi["muscle_glycogen"]]),
        float(state[mi["hepatic_output"]]),
        float(ra),
        float(act),
    )
    mg = fl["mg_dl_per_g"]
    kg = params.body_mass_kg / 1000.0
    uptake = fl["uptake_ii"] + fl["uptake_id"] + fl["uptake_ex"]
    store = fl["syn_M_id"]
    ra_g = fl["ra"] / mg
    uptake_g = uptake / mg
    booked = (
        (fl["ra"] - uptake) / mg
        + (fl["gng_rel_flux"] + fl["gng_divert"]) * kg
        - fl["brk_M_g"]
        + store
    )
    pools = fl["dG"] / mg + fl["dLGly"] + fl["dMGly"]
    return {
        "ra": ra_g,
        "syn_L": fl["syn_L"] / mg,
        "syn_M": fl["syn_M"] / mg + store,
        "glyco": fl["glyco_flux"] * kg,
        "gng_rel": fl["gng_rel_flux"] * kg,
        "gng_divert": fl["gng_divert"] * kg,
        "uptake": uptake_g - store,
        "brk_M": fl["brk_M_g"],
        "oxidized": uptake_g - store + fl["brk_M_g"],
        "residual": pools - booked,
    }


def _bile_flux(params: PatientParams, state: np.ndarray, duo: dict[str, float]) -> dict[str, float]:
    mi = MARKER_INDEX
    fl = bile_fluxes(
        params,
        float(state[mi["cck"]]),
        float(state[mi["gallbladder_bile"]]),
        float(state[mi["intestinal_bile"]]),
        float(state[mi["bile_acids"]]),
        duo["fat"], duo["protein"], duo["carb"],
    )
    return {
        "empty": fl["empty"],
        "ileal": fl["ileal"],
        "fill": fl["fill"],
        "direct": fl["direct"],
        "spill": fl["spill"],
        "serum_return": fl["serum_return"],
        "faecal": fl["faecal"],
        "synth": fl["synth"],
        "residual": fl["residual"],
    }


def _flux_pack(d: dict[str, float]) -> dict[str, float]:
    return {k: _r(v, 6) for k, v in d.items()}


def build_run(protocol_id: str, sample_every: int = SAMPLE_EVERY_MIN) -> dict[str, Any]:
    spec = _protocol_specs()[protocol_id]
    duration_min = int(spec["duration_min"])
    start_hour = float(spec["start_hour"])
    meals = [(float(a), float(b), float(c), float(d)) for a, b, c, d in spec["meals"]]
    sleep_wake = _sleep_mask(duration_min, start_hour)
    activity = np.zeros(duration_min, dtype=np.float32)
    walk = spec["walk"]
    if walk is not None:
        s, dur, intensity = walk
        activity[s:s + dur] = float(intensity)
        activity *= sleep_wake  # rest is 0 while asleep

    params = resolve_derived_params(PatientParams())
    traj, absorption = simulate_full_body(
        params,
        meals,
        sleep_wake,
        activity,
        duration_min,
        start_hour=start_hour,
        noise_scale=0.0,
        rng=np.random.default_rng(42),
    )

    times = list(range(0, duration_min, sample_every))
    n_m = len(MARKERS)
    center = np.asarray(NORM_CENTER, dtype=np.float64)
    scale = np.asarray(NORM_SCALE, dtype=np.float64)
    phys_lo = np.asarray(PHYSIOLOGICAL_MIN, dtype=np.float64)
    phys_hi = np.asarray(PHYSIOLOGICAL_MAX, dtype=np.float64)

    frames = []
    for t in times:
        state = traj[t].astype(np.float64)
        if t == 0:
            rate = np.zeros(n_m, dtype=np.float64)
        else:
            prev = traj[max(0, t - sample_every)].astype(np.float64)
            rate = (state - prev) / float(sample_every)
        z = (state - center) / scale
        gut = absorption[t]
        clamped = [
            MARKERS[i].id
            for i in range(n_m)
            if state[i] <= phys_lo[i] + 1e-6 or state[i] >= phys_hi[i] - 1e-6
        ]
        appearing = float(gut[3]) > 0.5
        duo = duodenal_delivery(float(t), meals, params)
        frames.append({
            "t": t,
            "clock_min": _r(_clock_minutes(start_hour, t), 1),
            "sleep_wake": _r(float(sleep_wake[t]), 3),
            "activity": _r(float(activity[t]), 3),
            "state": [_r(x) for x in state],
            "z": [_r(x, 3) for x in z],
            "rate": [_r(x, 5) for x in rate],
            "gut": [_r(x, 5) for x in gut],
            "duo": [_r(duo["fat"], 5), _r(duo["protein"], 5), _r(duo["carb"], 5)],
            "appearing": appearing,
            "clamped": clamped,
            "flux": {
                "carbon": _flux_pack(_carbon_flux(params, state, float(gut[0]), float(activity[t]))),
                "bile": _flux_pack(_bile_flux(params, state, duo)),
            },
        })

    meal_t = meals[0][0] if meals else None
    phases = dietary_carb_flow_phases_for_ui(duration_min, meal_t) if protocol_id == "meal" else []

    return {
        "schema": "pulse.lab.run.v2",
        "id": protocol_id,
        "title": spec["title"],
        "blurb": spec["blurb"],
        "source": "teacher",
        "patient": "default",
        "duration_min": duration_min,
        "start_hour": start_hour,
        "sample_every_min": sample_every,
        "marker_ids": [m.id for m in MARKERS],
        "gut_channel_ids": list(GUT_CHANNEL_IDS),
        "duodenal_channel_ids": list(DUODENAL_CHANNEL_IDS),
        "meals": [
            {"t": _r(t), "carbs": _r(c), "fats": _r(f), "proteins": _r(p)}
            for t, c, f, p in meals
        ],
        "phases": phases,
        "frames": frames,
    }


def write_lab(lab_dir: Path | None = None) -> Path:
    root = lab_dir or LAB_DIR
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    graph = build_graph()
    (root / "graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    index = []
    for pid in _protocol_specs():
        run = build_run(pid)
        (runs_dir / f"{pid}.json").write_text(json.dumps(run) + "\n")
        index.append({
            "id": run["id"],
            "title": run["title"],
            "blurb": run["blurb"],
            "duration_min": run["duration_min"],
            "file": f"runs/{pid}.json",
        })
    (root / "runs.json").write_text(json.dumps(index, indent=2) + "\n")
    return root
