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
| `[REFLEX]` | Clamp, default input, derived feeling |
| Identity color | Which embedding / which patient |
| Lagrangian tracer | A labeled parcel on a conserved loop (this plate, this dye) |

## Surfaces (increments)

1. **Coupling graph.** Modules as nodes, couplings as signed edges,
   markers as satellites when a module is selected. Time slider and
   playback of a teacher trajectory. Explode pulls modules apart so
   edges are readable. Right-hand readout of physical units. Protocol
   buttons (day, fast, dawn, meal). Badges for asleep, meal appearing,
   clamp, fasting. Graph schema is generated from `types.py`, so a
   layout change in the model is a layout change in the picture.
2. **Flux weather.** Carbon and bile are conserved
   loops on the same graph. Appearance travels gut → glucose; oxidation
   and faecal loss are drawn as holes (open rings). A ledger residual
   that fails to close is a leak badge — the teacher should not fire
   it; a later student might. Layer toggles: Coupling, Carbon,
   Bile. Duodenal delivery lights `duodenal.*` edges. Fluxes come from
   `glucose_fluxes` / `bile_fluxes`. `simulate_full_body` still returns
   `(traj, absorption)`.
3. **Live body (this increment).** One student simulation, interactive.
   Eat / Walk / Lie down write meals, activity, and sleep at *now*. Play
   steps the model forward. Nothing past now is computed or shown.
   Feelings on the readout (hungry, warm/cold, tired, sleepy) are derived
   from markers, not extra ODE states. Sleepy is clock + cortisol; Asleep
   is an input you set. The teacher is not this surface; accuracy of the
   student is a separate matter. Pass `--model path.pt` when a 30-D
   checkpoint exists; until then the lab runs the current architecture
   untrained.
4. **Tracers (next).** Lagrangian layer on the same carbon and bile
   loops. Eulerian weather stays: nodes are pools, edges are fluxes.
   Tracers ask where *this* parcel goes. Not a new ODE and not identity
   the student owns — pathlines of the live flux field, like CFD on a
   velocity snapshot. Physiological minutes are the clock.
5. **Anatomical stage (optional).** A dozen BodyParts3D organs, Pulse
   system layers, nodes parented to meshes. CC BY attribution required.
   Adult male reference only — say so on the about sheet. The graph,
   tracers, and telemetry do not change; tracers parent to the same
   waypoint coordinates.
6. **Agent tools.** `find_marker`, `inspect_module`, `play_protocol`,
   `isolate_pathway` — the atlas WebMCP pattern, aimed at this inspector.
7. **Not this surface.** Consumer 3D body, 2,234 selectable meshes,
   Minecraft, real-time 50 ms, ballistic dots along anatomy. Physiological
   minutes are the clock. Play speeds them up. Do not skip physiological
   time.

## Tracers (Lagrangian)

The graph is Eulerian. A tracer is the dual: follow a labeled parcel.

Pulse is thirty well-mixed markers plus a gut kernel. It does not
simulate particles. A tracer layer visualizes the fluxes already on the
loop (`glucose_fluxes` / `bile_fluxes`, student gut appearance). Same
waypoints, holes, and inlets.

**Compartmental, not a conveyor.** A pool is mixed. A marked glucose
does not march through blood in a line. It sits in the glucose node with
residence time ~ pool / outflow, then branches with probabilities
proportional to the live fluxes (appearance vs liver glycogen vs
oxidation; emptying vs ileal vs faecal). Holes are deaths (oxidized,
faecal). Inlets are births (appearance, GNG, bile synthesis).

**Spawn on appearance, not on Eat.** The gut kernel is already
Lagrangian in time: appearance is minutes-since-meal. The plate is not
in plasma at the click. Particles are born on `ra` / duodenal delivery
as the meal appears.

**Demo.** Eat a plate → a cloud at gut → drip into glucose along the
appearance curve → some park in liver glycogen, some burn at the
oxidized hole; a walk pulls more toward muscle oxidation. Same
machinery for a dye in the glucose pool (marked glucose, not a meal) and
for an arbitrary bolus (carb / fat / protein / bile as species on the
loops). Carbon is the one you feel. Bile is the prettier closed
circulation.

**Honesty.** Particles ≠ molecules. No ballistic fly-along-anatomy. No
50 ms game loop. If we are decorating rather than tracking identity the
student owns, say so (the `[REFLEX]` pattern). Eulerian weather stays
on; tracers are a toggle, not a replacement.

## Telemetry

`/simulate` today returns `series[t, marker]` plus a carb-flow side
channel. The lab packet is state, z, rates, gut appearance, duodenal
delivery, carbon and bile fluxes, sleep/activity, meals, clamps, and
derived feelings. The viewer talks to a local student session through
`integrate()`. Play is the clock. Tracers are a viewer of the flux
field, not extra student state.

## How to look

From the repo root:

```
uv run python scripts/lab_viewer.py
```

Optional checkpoint: `uv run python scripts/lab_viewer.py --model path/to/model.pt`.

Open http://127.0.0.1:8765. Eat, Walk, and Lie down poke the running
student at now. Play lives the next physiological minutes. Starting
scenes (Morning, Day, Fast, …) reset to t=0. `graph.json` is regenerated
on boot from live types. Do not commit `*.pt`.
