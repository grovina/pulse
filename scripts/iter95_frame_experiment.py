"""Does the raw-concentration frame remove BOTH the absorbing floor AND the iter-51
softplus-saturation trap?

Both claims are closed-form, so this tests the EQUILIBRIUM directly rather than rolling
out 400 Euler steps per optimizer step. With prod_scale = cons_scale*typical:

  DEVIATION (current):  0 = P*ps - C*cs*(raw-typ)/scale  =>  raw* = typ + scale*typ*P/C
                        P,C >= 0  =>  raw* >= typ  ALWAYS. That is the floor, exactly.
                        And raw* == typ requires P == 0, i.e. prod_logit -> -inf:
                        the iter-51 saturation trap.

  CONCENTRATION (fix):  0 = P*ps - C*cs*raw          =>  raw* = typ*P/C
                        reachable anywhere in (0, inf), and raw* == typ requires
                        P == C -- two moderate positive values, no saturation.

The optimizer is given a head and asked to place the equilibrium at a target. Reported:
achieved level; prod_logit (the PRE-softplus logit); and sigmoid(prod_logit), which is
d(softplus)/d(prod_logit) -- the fraction of gradient still reaching the head's MLP.
"""
import sys
from pathlib import Path
_ROOT = Path("/Users/grovina/Projects/grovina/pulse"); sys.path.insert(0, str(_ROOT))
import torch
import torch.nn.functional as F
from pulse.modules.base import SpeciesHead

TYPICAL, SCALE = 10.0, 10.0          # insulin's typical and NORM_SCALE
INPUT_DIM, HIDDEN, STEPS = 8, 16, 4000
X = torch.randn(INPUT_DIM, generator=torch.Generator().manual_seed(1)) * 0.3


def equilibrium(prod, cons, mode):
    """Closed-form equilibrium raw level. cons_scale cancels out of both."""
    ratio = prod / cons.clamp_min(1e-9)
    if mode == "deviation":
        return TYPICAL + SCALE * TYPICAL * ratio
    return TYPICAL * ratio


def fit(mode, target):
    torch.manual_seed(0)
    head = SpeciesHead(INPUT_DIM, HIDDEN)
    opt = torch.optim.Adam(head.parameters(), lr=0.02)
    for _ in range(STEPS):
        opt.zero_grad()
        raw_out = head.network(X)
        prod, cons = F.softplus(raw_out[0]), F.softplus(raw_out[1])
        loss = (equilibrium(prod, cons, mode) - target) ** 2
        loss.backward()
        opt.step()
    with torch.no_grad():
        raw_out = head.network(X)
        prod, cons = F.softplus(raw_out[0]), F.softplus(raw_out[1])
        return (float(equilibrium(prod, cons, mode)), float(raw_out[0]),
                float(torch.sigmoid(raw_out[0])))


print(f"one species, typical = {TYPICAL}, NORM_SCALE = {SCALE}  (insulin's)")
print("target 3.8 = teacher's 24h-fast insulin; 10 = typical; 50 = a meal peak\n")
print(f"{'assembly':<16}{'target':>8}{'achieved':>10}{'error':>9}"
      f"{'prod_logit':>12}{'grad frac':>11}  verdict")
for target in (3.8, 10.0, 50.0):
    for mode in ("deviation", "concentration"):
        got, logit, gfrac = fit(mode, target)
        if abs(got - target) > 0.05 * max(target, 1.0):
            verdict = "**CANNOT REACH — floored at typical**"
        elif gfrac < 1e-3:
            verdict = "**reached, but softplus SATURATED**"
        else:
            verdict = "ok"
        print(f"{mode:<16}{target:>8.2f}{got:>10.3f}{got - target:>9.3f}"
              f"{logit:>12.3f}{gfrac:>11.2e}  {verdict}")
    print()
