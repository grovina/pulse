"""iter-80 post-mortem: model vs teacher glucose over the sleep-48h-adequate protocol.

Answers the fork: on the teacher cohort episode that blew up (teacher-source MAPE 6.95),
is it the MODEL prediction that diverges, or the TEACHER reference (mech-2 distortion)?
"""
import sys
import numpy as np
import torch

from pulse.knowledge.benchmark_extras import cohort_sleep_48h_benchmark_episodes
from pulse.knowledge.full_body import PatientParams, simulate_full_body
from pulse.knowledge.cohorts.sleep import (
    SLEEP_COHORT_48H_MEALS, SLEEP_COHORT_48H_START_HOUR, sleep_two_nights_adequate,
)
from pulse.diagnostics.probe import load_model_from_checkpoint
from pulse.model import integrate
from pulse.benchmark import calibrate_embedding
from pulse.types import MARKER_INDEX
from pulse.modules.gut import MealEvent

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/iter80_model.pt"
GI, BI = MARKER_INDEX["glucose"], MARKER_INDEX["bhb"]


def stats(name, arr):
    a = np.asarray(arr, dtype=float)
    print(f"  {name:28s} min={a.min():8.2f}  max={a.max():8.2f}  "
          f"end={a[-1]:8.2f}  mean={a.mean():8.2f}  nan={np.isnan(a).any()}")


# ---- protocol ----
dur = 2880
start_hour = SLEEP_COHORT_48H_START_HOUR
meals_raw = list(SLEEP_COHORT_48H_MEALS)
sw = np.asarray(sleep_two_nights_adequate(dur, start_hour), dtype=np.float32).reshape(-1)[:dur]
act = np.full(dur, 0.05, dtype=np.float32)

# ---- grab the real benchmark episode (initial state + calibration check-ins) ----
eps = cohort_sleep_48h_benchmark_episodes()
ep = eps[0]
print(f"episode: {ep.user_id}  source={ep.source}  dur={ep.duration_min}  "
      f"n_checkins={len(ep.calibration_check_ins)}  n_eval={len(ep.eval_measurements)}")
init_state = np.asarray(ep.initial_state, dtype=np.float32)

# ---- TEACHER reference (iter-80 params) ----
traj80, _ = simulate_full_body(PatientParams(), meals_raw, sw, act,
                               dur, start_hour, noise_scale=0.0,
                               rng=np.random.default_rng(0))
traj80 = np.asarray(traj80)
print("\n[TEACHER iter-80]")
stats("glucose (mg/dL)", traj80[:, GI])
stats("bhb (mM)", traj80[:, BI])

# ---- TEACHER with mechanism-2 effectively OFF (glycogenolytic share -> full, no keto ramp) ----
# Approximate the pre-iter80 teacher: keep all hepatic output glycogenolytic-equivalent
# and disable the glycogen->ketosis gain, isolating mech-2's effect on the 48h reference.
p_off = PatientParams()
p_off.hep_glyco_frac = 0.0   # no availability-scaled drawdown -> output not gated by depletion
p_off.hep_gng_comp = 1.0     # full compensation
p_off.keto_glyc_gain = 0.0   # no glycogen->ketosis ramp
trajoff, _ = simulate_full_body(p_off, meals_raw, sw, act,
                                dur, start_hour, noise_scale=0.0,
                                rng=np.random.default_rng(0))
trajoff = np.asarray(trajoff)
print("\n[TEACHER mech-2 OFF]")
stats("glucose (mg/dL)", trajoff[:, GI])
stats("bhb (mM)", trajoff[:, BI])

# ---- MODEL prediction (calibrated embedding on real check-ins) ----
model, _ = load_model_from_checkpoint(CKPT)
model.eval()
meal_events = [MealEvent(time=float(t), carbs=float(c), fats=float(f), proteins=float(p))
               for t, c, f, p in meals_raw]
init_t = torch.tensor(init_state, dtype=torch.float32)
sw_t = torch.tensor(sw, dtype=torch.float32)
act_t = torch.tensor(act, dtype=torch.float32)

# build calibration observations from the episode's own check-ins
import time
CAL_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 64

emb_dim = model.embedding_prior_mean.numel() if hasattr(model, "embedding_prior_mean") else 64
zero_emb = torch.zeros(emb_dim)
t0 = time.time()
with torch.no_grad():
    pred0 = integrate(model=model, initial_state=init_t, embedding=zero_emb,
                      n_steps=dur, dt=1.0, start_time_minutes=start_hour * 60.0,
                      meals=meal_events, sleep_wake=sw_t, activity=act_t).numpy()
print(f"\n[MODEL iter-80, ZERO embedding] (rollout {time.time()-t0:.1f}s)")
stats("glucose (mg/dL)", pred0[:, GI])
stats("bhb (mM)", pred0[:, BI])

from pulse.benchmark import measurement_points_from_check_ins  # noqa
cal_obs = measurement_points_from_check_ins(ep.calibration_check_ins, dur)
t0 = time.time()
cal = calibrate_embedding(
    model=model, observations=cal_obs, initial_state=init_t,
    meals=meal_events, duration_min=dur, start_time_minutes=start_hour * 60.0,
    n_steps=CAL_STEPS, lr=0.05, l2_weight=0.003, sleep_wake=sw_t, activity=act_t,
)
print(f"\ncalibration ({CAL_STEPS} steps, {time.time()-t0:.1f}s) final_loss={cal.final_loss:.4f}")
with torch.no_grad():
    pred = integrate(model=model, initial_state=init_t, embedding=cal.embedding,
                     n_steps=dur, dt=1.0, start_time_minutes=start_hour * 60.0,
                     meals=meal_events, sleep_wake=sw_t, activity=act_t).numpy()
print(f"\n[MODEL iter-80 (calibrated {CAL_STEPS})]")
stats("glucose (mg/dL)", pred[:, GI])
stats("bhb (mM)", pred[:, BI])

# ---- eval-point MAPE: model vs teacher-iter80 reference ----
mg, tg = pred[:, GI], traj80[:, GI]
mape = np.mean(np.abs(mg - tg) / np.maximum(np.abs(tg), 1e-6))
print(f"\nglucose MAPE model-vs-teacher80 (dense) = {mape:.3f}")
# where does it blow up?
err = np.abs(mg - tg) / np.maximum(np.abs(tg), 1e-6)
for h in (6, 12, 18, 24, 30, 36, 42, 47):
    i = min(h * 60, dur - 1)
    print(f"  t={h:2d}h  teacher={tg[i]:7.2f}  model={mg[i]:7.2f}  ape={err[i]:6.2f}")
