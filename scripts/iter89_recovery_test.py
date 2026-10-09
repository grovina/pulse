"""Calibration-recovery test: can we actually FIND a person?

Feynman-honest setup. We generate synthetic people from the LEARNED prior, simulate
their observed markers with the model ITSELF (so there is zero model error - this is
the BEST CASE for identifiability), then run the REAL benchmark calibration and ask:

  does the recovered embedding reproduce the person's true physiology,
  and does it beat the trivial "just use the population mean" baseline?

If calibration cannot beat prior_mean here, it is not identifying individuals at all.

We also track HRV0, which is NOT observed - it tests the claim that off-axis
physiology is ill-posed and gets dragged along spurious entanglement.

Si is the metabolic module's person_params['si'] (student frame: per normalized
insulin unit, 10x the teacher's Si). It is not a marker; glucose is what identifies it.
"""
import sys, time
from pathlib import Path

# An editable install can shadow this tree (amplification_probe.py). The checkpoint's
# architecture is this repo's, so the probe has to import this repo.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch, numpy as np
import pulse
assert Path(pulse.__file__).resolve().is_relative_to(_ROOT), f"shadowed pulse: {pulse.__file__}"

from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.types import MARKER_INDEX, NORM_CENTER, EMBEDDING_DIM
from pulse.benchmark import calibrate_embedding, MeasurementPoint
from pulse.modules.gut import MealEvent
from pulse.modules import metabolic as M
from pulse.modules import cardiovascular as C

N_PEOPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
CKPT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/iter111.pt"
# Trimmed vs the benchmark's 720/504 so this is tractable on a 4GB CPU box
# (calibration costs ~1 full rollout per Adam step). Same structure, shorter horizon.
DURATION = 420
CAL_END = 294
WINDOW = 180

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
m = ModularPhysiologyNetwork.from_checkpoint(ck); m.eval()
pm = torch.tensor(ck['embedding_prior_mean']); ps = torch.tensor(ck['embedding_prior_std'])
print(f'checkpoint {CKPT}  people={N_PEOPLE}  adam_steps={N_STEPS}  duration={DURATION}')

IDX = {k: MARKER_INDEX[k] for k in ('glucose','hr','hrv','sbp','dbp','temp')}

def physiology(emb):
    """The per-patient physiology the embedding encodes (the thing to recover)."""
    with torch.no_grad():
        e_met = m.embedding_projections['metabolic'](emb)
        e_cvs = m.embedding_projections['cardiovascular'](emb)
        # Decoded by the modules themselves: A1 (PLAN.md) moved Gb and the vitals
        # into log space, so any hand-rebuilt `MAX_Z * tanh(head)` is now wrong.
        Gb = float(m.metabolic.glucose_setpoint_raw(e_met))
        Si = float(m.metabolic.person_params(e_met)['si'])
        sp_raw = m.cardiovascular.setpoints_raw(e_cvs)
        out = {'Gb': Gb, 'Si': Si}
        for j, k in enumerate(('hr','hrv','sbp','dbp')):
            out[k.upper()+'0'] = float(sp_raw[j])
    return out

# meals in the calibration window only (mirrors the benchmark episodes)
MEALS = [MealEvent(time=90.0, carbs=60.0, fats=10.0, proteins=15.0)]
# sparse multi-marker check-ins, calibration window only. hrv is NOT observed.
OBS_PLAN = {'glucose':60, 'hr':60, 'sbp':120, 'dbp':120, 'temp':120}

def person_initial_state(phys):
    st = torch.tensor(NORM_CENTER, dtype=torch.float32).clone()
    st[IDX['glucose']] = phys['Gb']; st[IDX['hr']] = phys['HR0']
    st[IDX['hrv']] = phys['HRV0']; st[IDX['sbp']] = phys['SBP0']; st[IDX['dbp']] = phys['DBP0']
    return st

def err(a, b, keys):
    return {k: abs(a[k]-b[k]) for k in keys}

def num(x):
    """Gb and the vitals are O(10-100); Si is O(1e-3). Don't round Si to 0.00."""
    ax = abs(float(x))
    return f'{x:.2f}' if ax == 0 or ax >= 0.05 else f'{x:.4g}'

KEYS = ['Gb','Si','HR0','SBP0','DBP0','HRV0']
torch.manual_seed(7)
rows = []
prior_phys = physiology(pm)

for i in range(N_PEOPLE):
    emb_true = pm + ps * torch.randn(EMBEDDING_DIM)
    true_phys = physiology(emb_true)
    st0 = person_initial_state(true_phys)

    # simulate the person's TRUE trajectory with the model itself (no model error)
    with torch.no_grad():
        traj = integrate(m, st0, emb_true, n_steps=DURATION, dt=1.0, meals=MEALS)

    obs = [MeasurementPoint(time=t, marker_id=mk, value=float(traj[t, IDX[mk]]))
           for mk, iv in OBS_PLAN.items() for t in range(0, CAL_END, iv)]

    t0 = time.time()
    res = calibrate_embedding(
        m, obs, initial_state=st0, meals=MEALS, duration_min=DURATION,
        start_time_minutes=360.0, n_steps=N_STEPS, lr=0.05, window_size=WINDOW,
        l2_weight=0.003, prior_mean=pm, prior_std=ps, prior_weight=0.0, max_norm=3.0,
    )
    rec_phys = physiology(res.embedding)

    e_cal = err(rec_phys, true_phys, KEYS)
    e_pri = err(prior_phys, true_phys, KEYS)
    cos = float(torch.nn.functional.cosine_similarity(res.embedding, emb_true, dim=0))
    rows.append((e_cal, e_pri, cos, float((res.embedding-emb_true).norm()),
                 float(res.embedding.norm()), float(emb_true.norm())))
    print(f'person {i+1}/{N_PEOPLE}  ({time.time()-t0:.0f}s)  loss={res.final_loss:.4f}  '
          f'cos(rec,true)={cos:+.3f}  ||rec||={res.embedding.norm():.2f} ||true||={emb_true.norm():.2f}')
    for k in KEYS:
        obs_tag = '' if k != 'HRV0' else '  [UNOBSERVED]'
        print(f'   {k:5} true {num(true_phys[k]):>10} | calib {num(rec_phys[k]):>10} (err {num(e_cal[k]):>10})'
              f' | prior-mean {num(prior_phys[k]):>10} (err {num(e_pri[k]):>10}){obs_tag}')

print('\n================ SUMMARY: does calibration beat the population mean? ================')
print(f'{"marker":8} {"mean |err| calib":>17} {"mean |err| prior":>17} {"skill (1-c/p)":>14}')
for k in KEYS:
    c = np.mean([r[0][k] for r in rows]); p = np.mean([r[1][k] for r in rows])
    skill = 1 - c/p if p > 1e-9 else float('nan')
    tag = '  <- UNOBSERVED' if k == 'HRV0' else ''
    print(f'{k:8} {num(c):>17} {num(p):>17} {skill:13.2f}{tag}')
print(f'\nmean cos(recovered, true embedding) = {np.mean([r[2] for r in rows]):+.3f}  '
      f'(1.0 = embedding itself recovered; ~0 = different code, same physiology)')
print(f'mean ||recovered - true|| = {np.mean([r[3] for r in rows]):.2f}   '
      f'(mean ||true||={np.mean([r[5] for r in rows]):.2f})')
