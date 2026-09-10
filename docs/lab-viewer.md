# Lab viewer

A picture of Pulse as a living coupling graph. Lab surface, not the consumer
app: this is how we look at the whole system. The consumer surface stays
check-ins and feelings.

The graph is the first increment. Anatomical meshes are an optional later
layer on the same nodes, not a different product.

## Why a graph first

Pulse is already a graph. `MODULE_COUPLING_CHANNELS` in `pulse/types.py` is
what `forward()` reads. Eight nodes (gut kernel + seven ODE modules), thirty
markers, a handful of appearance/delivery channels. That is the whole
runtime. A 2,234-piece cadaver is the wrong resolution for a 30-D ODE; a
pure chart dump never shows that insulin left metabolic and arrived at
appetite.

A graph with **fixed body-ish positions** (stress at the head, heart and
lungs in the chest, gut/liver/bile in the abdomen) is a step toward anatomy
without committing to BodyParts3D. When we want organs, those node
coordinates become attachment points on meshes. Edges, playback, badges,
and telemetry stay the same.

## What two outside repos contributed

**Human Atlas** (ashemag/human-atlas) is a catalog of structure: BodyParts3D
meshes, system layers, explode, isolate, search, WebMCP inspect tools. GPU
state textures keep thousands of pieces interactive. Nothing in it moves
except the camera. Steal: layers, explode, isolate, concept≠mesh,
agent inspect. Do not steal the full catalog as the product.

**Fly Brain Minecraft** (blendi-remade/fly-brain-minecraft) is a live
readout of activity on a graph. Split HUD (spatial map + neuroscope), heat
that decays on a physiological time constant, identity tying body to panel,
telemetry as a first-class packet, canned stimuli (`feed`, `loom`), and a
`[REFLEX]` badge when scaffolding — not the wiring — is driving. Steal:
weather on the graph, dual HUD, honest badges, protocols as buttons,
engine-independent telemetry. Do not steal Minecraft, uniform cells, or a
50 ms game loop.

Pulse sits between them: atlas has the stage, the fly has the weather. We
have the weather and the graph already. The stage can wait.

## Mapping

| Their object | Pulse object |
|---|---|
| Anatomical system | `System` module + gut kernel |
| Named concept (“liver”) | Markers and couplings that live there |
| Connectome edge | `MODULE_COUPLING_CHANNELS` |
| Spike | `|dx/dt|` or a named flux |
| Motor channel | Readouts a person would notice (glucose, HR, ghrelin, …) |
| Sensory frame | Meals, sleep, activity, check-ins |
| `[REFLEX]` | Clamp, default input, rejected calibration, teacher overlay |
| Identity color | Which embedding / which patient |

## Surfaces (increments)

1. **Coupling graph.** Modules as nodes, couplings as signed edges,
   markers as satellites when a module is selected. Time slider and
   playback of a teacher trajectory. Explode pulls modules apart so
   edges are readable. Right-hand readout of physical units. Protocol
   buttons (day, fast, dawn, meal). Badges for asleep, meal appearing,
   clamp, fasting. Graph schema is generated from `types.py`, so a
   layout change in the model is a layout change in the picture.
2. **Flux weather (this increment).** Carbon and bile are conserved
   loops on the same graph. Appearance travels gut → glucose; oxidation
   and faecal loss are drawn as holes (open rings). A ledger residual
   that fails to close is a leak badge — the teacher should not fire
   it; a later student overlay will. Layer toggles: Coupling, Carbon,
   Bile. Duodenal delivery lights `duodenal.*` edges. Fluxes are
   reconstructed each frame from `glucose_fluxes` / `bile_fluxes` plus
   the trajectory; `simulate_full_body` still returns `(traj, absorption)`.
3. **Student overlay.** Same graph, teacher vs student vs prior-mean
   embedding. Construction pins as visible properties (SpO2 cannot paint
   outside (70, 100); bile is a loop or the loop is drawn broken).
4. **Anatomical stage (optional).** A dozen BodyParts3D organs, Pulse
   system layers, nodes parented to meshes. CC BY attribution required.
   Adult male reference only — say so on the about sheet. The graph and
   telemetry do not change.
5. **Agent tools.** `find_marker`, `inspect_module`, `play_protocol`,
   `isolate_pathway` — the atlas WebMCP pattern, aimed at this inspector.
6. **Not this surface.** Consumer 3D body, 2,234 selectable meshes,
   Minecraft, real-time 50 ms. Playback of a finished integrate is the
   clock. Do not skip physiological time.

## Telemetry

`/simulate` today returns `series[t, marker]` plus a carb-flow side
channel. The lab run file is the proto-packet: state, z, rates, gut
appearance, duodenal delivery, carbon and bile fluxes, sleep/activity,
meals, which clamps fired. The viewer consumes files; it does not own
`integrate()`. Later the engine can stream the same shape.

## How to look

From the repo root, after generating runs:

```
uv run python scripts/export_lab_viewer.py
uv run python -m http.server 8765 --directory lab
```

Open http://localhost:8765. Regenerating overwrites `lab/graph.json` and
`lab/runs/*.json` from the live types and teacher. The committed copies
are the last exported snapshot so the page works without a sim.
