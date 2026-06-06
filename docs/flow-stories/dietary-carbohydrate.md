# Flow story: dietary carbohydrate (logged mixed meal)

This document is the **contract** for the first end-to-end “nutrient flow” story in Pulse: not literal molecule tracking, but a **canonical packet of carbohydrate** from a logged meal through pools and signals the engine already represents.

The operational acceptance criteria live in code: `dietary_carbohydrate_meal_flow_scenario` in `pulse/knowledge/textbook_scenarios/flow_stories.py`, run via `run_all_scenarios()` with the default cold (full-body teacher) trajectory.

**Neural model:** from `pulse/knowledge/textbook_scenarios/neural_rollout.py` build `make_dietary_carbohydrate_neural_trajectory_fn(model, embedding)`, then either:

- pass it only to this scenario’s runner, or
- use `run_all_scenarios(trajectory_fn_by_name={"dietary_carbohydrate_meal_flow_scenario": traj_fn})` so OGTT and other textbook scenarios **keep** their cold trajectories (they ignore the hook).

Passing a single global `trajectory_fn` into `run_all_scenarios` is reserved for when every runner understands the same provider; today only the dietary-carbohydrate flow story consumes it.

That path integrates `ModularPhysiologyNetwork` for this protocol while injecting **per-minute gut outputs from the cold teacher**, so checks that use glucose appearance (and lipid/amino flags) stay aligned with the teacher’s absorption curve without double-counting the learned Gut module.

### Product surface

The Pulse engine `/simulate` response includes **`carb_flow`**: sampled times, **glucose appearance** from the learned `GutModule`, selected marker series, **meal_time_min** (first logged meal with meaningful carbs in that window), and **phases** for UI bands. Phase **minute windows** match `pulse/knowledge/textbook_scenarios/flow_story_protocol.py` (`dietary_carb_flow_phases_for_ui`): baseline and recovery are absolute; other bands shift with `meal_time_min` using the same offsets as the dietary-carbohydrate scenario checks. The Pulse app home screen exposes a **Carb flow** tab (alongside **By system**) using the same run as the marker grid.

## Scope (honest framing)

- **In scope:** gut-scale **glucose appearance** (from meal carbs), plasma **glucose**, **insulin** / **glucagon**, **FFA**, appetite hormones (**ghrelin**, **GLP-1**), **core temperature** (diet-induced thermogenesis enters through total nutrient appearance in the cold model), and **recovery** of glucose toward the pre-meal band over a long enough horizon.
- **Out of scope for this story:** atom-level calcium, explicit glycogen compartments, renal excretion, pharmacologic insulin.

## Protocol (simulation)

- **Duration:** 480 minutes (8 hours). Shorter windows are insufficient for the cold teacher to bring glucose back near baseline after a substantial carb load; the checks encode that reality.
- **Meal:** single mixed meal at **t = 30 min**: 50 g carbohydrate, 12 g fat, 18 g protein (tuple order matches `simulate_full_body`: time, carbs, fats, proteins).
- **Context:** awake throughout (`sleep_wake = 1`), light constant activity (`0.05`), morning start (`start_hour = 8.0`), default `PatientParams()`, low process noise.

## Narrative phases → observable checks

| Phase | Physiology (plain language) | What we assert on the cold trajectory |
|--------|-----------------------------|----------------------------------------|
| Baseline | Fasting-ish morning before meal | Pre-meal windows for means (indices `0 : DIETARY_CARB_PRE_MEAL_END`). |
| Appearance | Carbohydrate enters as **glucose appearance rate** | Peak **Ra** in `DIETARY_CARB_RA_WINDOW_*` relative to protocol meal at 30 min. |
| Excursion | Plasma glucose rises | Max glucose in `DIETARY_CARB_GLUCOSE_EXCURSION_*` exceeds pre-meal mean by a margin. |
| Endocrine response | Insulin responds to glycemia / incretin | Peak insulin in Ra window through `DIETARY_CARB_INSULIN_PEAK_SCAN_END` vs glucose peak timing. |
| Counter-regulatory tone | Glucagon relative to fasting | Mean glucagon in `DIETARY_CARB_GLUCAGON_POST_*` below pre-meal mean. |
| Lipid routing | Insulin antilipolysis | Mean FFA in `DIETARY_CARB_FFA_POST_*` below pre-meal mean. |
| Appetite signals | Ghrelin falls with feeding; GLP-1 pulses | Ghrelin mean in `DIETARY_CARB_GHRELIN_POST_*`; GLP-1 max in `DIETARY_CARB_GLP1_PULSE_*`. |
| Thermogenesis | Post-meal metabolic load → core temperature | Mean temperature in `DIETARY_CARB_TEMP_POST_*` above pre-meal mean. |
| Recovery | Glucose returns toward homeostatic setpoint | Mean glucose in `DIETARY_CARB_GLUCOSE_RECOVERY_*` near pre-meal mean. |

Constants live in `flow_story_protocol.py`; `flow_stories.py` imports them for checks. UI phases use the same windows via `dietary_carb_flow_phases_for_ui` (meal-relative bands shift when the logged meal is not at t=30).

## Using this story elsewhere

- **Training / verification:** treat the scenario checks as one more weak constraint on the teacher or on a trained model by passing a compatible `trajectory_fn` when that API is wired through.
- **Product (“flow view”):** the same check names and time windows can drive timeline annotations, as long as the UI only claims what the state vector contains.

## References (high level)

Postprandial glucose–insulin ordering, glucagon suppression, FFA suppression, ghrelin dynamics, GLP-1 response to nutrient delivery, and DIT are standard textbook physiology; quantitative timings here are **calibrated to the repository cold model**, not to a specific citation curve-by-curve.
