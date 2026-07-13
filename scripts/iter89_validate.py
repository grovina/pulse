"""Iter-89 pre-dispatch validation gauntlet.

Checks the four folded-in changes are correct and, critically, that the new
insulin_action state does NOT reintroduce the phase-2 long-rollout NaN that
crashed iters 83-85.
"""
import torch
from pulse.types import EMBEDDING_DIM
from pulse.types import STATE_DIM, MARKER_INDEX, NORM_CENTER, NORM_SCALE
from pulse.model import ModularPhysiologyNetwork, integrate, precompute_gut_outputs
from pulse.modules.gut import MealEvent

torch.manual_seed(0)
HR = MARKER_INDEX["hr"]; HRV = MARKER_INDEX["hrv"]
SBP = MARKER_INDEX["sbp"]; DBP = MARKER_INDEX["dbp"]
GLU = MARKER_INDEX["glucose"]; INS = MARKER_INDEX["insulin"]
IA = MARKER_INDEX["insulin_action"]
CVS = [HR, HRV, SBP, DBP]

m = ModularPhysiologyNetwork()
m.eval()
center = torch.tensor(NORM_CENTER, dtype=torch.float32)

# --- 1. Batched forward finite (with precomputed gut) ---
emb_b = torch.randn(4, EMBEDDING_DIM)
meals = [MealEvent(time=30.0, carbs=60.0, fats=5.0, proteins=10.0)]
go = precompute_gut_outputs(m, emb_b, 60, meals=meals)
stb = center.unsqueeze(0).repeat(4, 1)
rb = m(stb, emb_b, torch.tensor(360.0), meals, gut_override=go[:, 0])
print("1. batched forward finite:", torch.isfinite(rb).all().item())

# --- 2. CVS setpoint moves each vital's equilibrium monotonically ---
# Force the setpoint head bias to +/- 1 (post-tanh ~ +/-0.76 * MAX_Z) and roll
# out a no-meal fast; the resting level of each vital should shift up vs down.
def resting(bias_val):
    mm = ModularPhysiologyNetwork(); mm.eval()
    with torch.no_grad():
        mm.cardiovascular.setpoint_net[-1].bias.copy_(torch.full((4,), float(bias_val)))
    emb = torch.zeros(EMBEDDING_DIM)
    traj = integrate(mm, center.clone(), emb, n_steps=600, dt=1.0)
    return traj[-1, CVS]

lo = resting(-1.0); hi = resting(+1.0)
print("2. CVS setpoint monotonic (hi>lo for all 4):",
      bool((hi > lo).all().item()), "| delta bpm/ms/mmHg:",
      [round(float(x), 2) for x in (hi - lo)])

# --- 3. insulin_action INVARIANT (untrained drift makes a rollout "fast"
#        meaningless, so check the rate law directly): at baseline insulin with
#        Xa=0 the state is at equilibrium (rate 0) and contributes nothing to
#        glucose clearance; raising insulin makes the rate strictly positive. ---
emb = torch.zeros(EMBEDDING_DIM)
st0 = center.clone()  # insulin == baseline -> relu(insulin_norm)=0; Xa == typical 0
r0 = m(st0, emb, torch.tensor(360.0), [])
ia_eq_rate = float(r0[IA].abs())
print("3. insulin_action rate at baseline eq (expect ~0):", round(ia_eq_rate, 6))
st_hi = center.clone(); st_hi[INS] = NORM_CENTER[INS] + 4 * NORM_SCALE[INS]
r_hi = m(st_hi, emb, torch.tensor(360.0), [])
print("   insulin_action rate when insulin high (expect >0):", round(float(r_hi[IA]), 4))
# Lag: after an insulin step, Xa rises slowly toward relu(insulin_norm)=4.0.
traj2 = integrate(m, st_hi, emb, n_steps=120, dt=1.0)
ia_series = traj2[:, IA]
print("   insulin_action rises & lags (t0,t10,t30,t60):",
      [round(float(ia_series[i]), 3) for i in (0, 10, 30, 60)])
ia_ok = ia_eq_rate < 1e-6 and float(r_hi[IA]) > 0

# --- 4. THE crash check: 24h rollout backward, zero non-finite grads ---
# Mirrors the phase-2 long rollout that NaN'd iters 83-85. Include meals + a
# calibratable embedding with grad.
emb_g = torch.randn(EMBEDDING_DIM, requires_grad=True)
meals24 = [MealEvent(time=float(t), carbs=75.0, fats=10.0, proteins=15.0)
           for t in (90, 510, 900)]
traj24 = integrate(m, center.clone(), emb_g, n_steps=1440, dt=1.0, meals=meals24,
                   checkpoint_segments=8)
loss = (traj24[:, [GLU, HR, SBP, DBP]] ** 2).mean()
loss.backward()
g = emb_g.grad
print("4. 24h rollout: state finite:", torch.isfinite(traj24).all().item(),
      "| loss finite:", torch.isfinite(loss).item(),
      "| grad finite:", torch.isfinite(g).all().item(),
      "| grad norm:", round(float(g.norm()), 4))

ok = (torch.isfinite(rb).all() and bool((hi > lo).all()) and ia_ok
      and torch.isfinite(traj24).all() and torch.isfinite(g).all())
print("\nGAUNTLET", "PASS" if ok else "FAIL")
