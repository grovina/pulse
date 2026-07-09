"""Controlled counterfactual on the SAME iter-89 network: does a physiological Sg
(and/or an enabled prior) restore identifiability? Only Sg and prior_weight vary."""
import sys, time, torch, numpy as np
import pulse.modules.metabolic as M

SG_KEFF   = float(sys.argv[1])   # desired RAW k_eff /min  (iter-89 trained = 0.00194)
PRIOR_W   = float(sys.argv[2])   # calibration prior weight (benchmark default 0.0)
N_PEOPLE  = int(sys.argv[3]); N_STEPS = int(sys.argv[4])

# old frame: k_eff = sg/30  ->  pin sg = k_eff*30, bypassing the band
M._SG_MIN = SG_KEFF * 30.0
M._SG_RANGE = 0.0

from pulse.model import ModularPhysiologyNetwork, integrate
from pulse.types import MARKER_INDEX, NORM_CENTER
from pulse.benchmark import calibrate_embedding, MeasurementPoint
from pulse.modules.gut import MealEvent
from pulse.modules import cardiovascular as C

ck = torch.load('/tmp/iter89.pt', map_location='cpu', weights_only=False)
m = ModularPhysiologyNetwork(embedding_dim=64, metabolic_hidden=48, cardiovascular_hidden=48,
    gut_hidden=32, appetite_hidden=24, stress_hidden=24, thermoreg_hidden=16, respiratory_hidden=16)
m.load_state_dict(ck['model_state']); m.eval()
pm = torch.tensor(ck['embedding_prior_mean']); ps = torch.tensor(ck['embedding_prior_std'])
IDX = {k: MARKER_INDEX[k] for k in ('glucose','hr','hrv','sbp','dbp','temp')}

def phys(emb):
    with torch.no_grad():
        em = m.embedding_projections['metabolic'](emb); ec = m.embedding_projections['cardiovascular'](emb)
        b = M._GLUCOSE_BASELINE_MAX_Z*torch.tanh(m.metabolic.glucose_baseline_net(em).squeeze(-1))
        sp = C._CVS_BASELINE_MAX_Z*torch.tanh(m.cardiovascular.setpoint_net(ec))
        sc = [10.,15.,10.,8.]
        o = {'Gb': 95.0+30.0*float(b)}
        for j,k in enumerate(('hr','hrv','sbp','dbp')): o[k.upper()+'0']=NORM_CENTER[IDX[k]]+sc[j]*float(sp[j])
    return o

MEALS=[MealEvent(time=90.0,carbs=60.0,fats=10.0,proteins=15.0)]
OBS={'glucose':60,'hr':60,'sbp':120,'dbp':120,'temp':120}
DUR,CAL,WIN=420,294,180
KEYS=['Gb','HR0','SBP0','DBP0','HRV0']
torch.manual_seed(7)
pp = phys(pm); rows=[]
sg_eff = M._SG_MIN
print(f'ARM: Sg_keff={SG_KEFF:.5f}/min (tau {1/SG_KEFF:.0f} min), prior_weight={PRIOR_W}')
for i in range(N_PEOPLE):
    et = pm + ps*torch.randn(64); tp = phys(et)
    st = torch.tensor(NORM_CENTER,dtype=torch.float32).clone()
    for k,mk in (('Gb','glucose'),('HR0','hr'),('HRV0','hrv'),('SBP0','sbp'),('DBP0','dbp')): st[IDX[mk]]=tp[k]
    with torch.no_grad(): tr = integrate(m, st, et, n_steps=DUR, dt=1.0, meals=MEALS)
    obs=[MeasurementPoint(time=t,marker_id=mk,value=float(tr[t,IDX[mk]])) for mk,iv in OBS.items() for t in range(0,CAL,iv)]
    t0=time.time()
    r = calibrate_embedding(m, obs, initial_state=st, meals=MEALS, duration_min=DUR,
        start_time_minutes=360.0, n_steps=N_STEPS, lr=0.05, window_size=WIN, l2_weight=0.003,
        prior_mean=pm, prior_std=ps, prior_weight=PRIOR_W, max_norm=3.0)
    rp = phys(r.embedding)
    rows.append(({k:abs(rp[k]-tp[k]) for k in KEYS},{k:abs(pp[k]-tp[k]) for k in KEYS}))
    print(f'  p{i+1} ({time.time()-t0:.0f}s) ||rec||={r.embedding.norm():.2f} '
          f'Gb true={tp["Gb"]:.1f} calib={rp["Gb"]:.1f} prior={pp["Gb"]:.1f}')
print(f'  {"marker":7} {"calib":>8} {"prior":>8} {"skill":>8}')
for k in KEYS:
    c=np.mean([r[0][k] for r in rows]); p=np.mean([r[1][k] for r in rows])
    print(f'  {k:7} {c:8.2f} {p:8.2f} {1-c/p:8.2f}')
