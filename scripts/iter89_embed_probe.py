"""Empirical probe of the iter-89 embedding->physiology map.

Answers: do per-patient embeddings self-organize into a coherent, physiologically
spanning, identifiable map that calibration can invert? We sample embeddings from
the LEARNED prior N(mean, std) and read the per-patient physiology the heads emit
(glucose baseline Gb, meal gain Ra, and the 4 cardiovascular setpoints), then look
at range, physiological plausibility, and cross-marker entanglement.
"""
import torch, numpy as np
from pulse.model import ModularPhysiologyNetwork
from pulse.types import MARKER_INDEX, NORM_CENTER, NORM_SCALE
from pulse.modules import metabolic as M
from pulse.modules import cardiovascular as C

ck = torch.load('/tmp/iter89.pt', map_location='cpu', weights_only=False)
m = ModularPhysiologyNetwork(
    embedding_dim=ck['embedding_dim'],
    metabolic_hidden=48, cardiovascular_hidden=48, gut_hidden=32,
    appetite_hidden=24, stress_hidden=24, thermoreg_hidden=16, respiratory_hidden=16,
)
m.load_state_dict(ck['model_state']); m.eval()
pm = torch.tensor(ck['embedding_prior_mean']); ps = torch.tensor(ck['embedding_prior_std'])

def physiology(emb):
    """emb [K,64] -> dict of per-patient physiological quantities the heads emit."""
    with torch.no_grad():
        e_met = m.embedding_projections['metabolic'](emb)
        e_cvs = m.embedding_projections['cardiovascular'](emb)
        b = M._GLUCOSE_BASELINE_MAX_Z * torch.tanh(m.metabolic.glucose_baseline_net(e_met).squeeze(-1))
        Gb = NORM_CENTER[MARKER_INDEX['glucose']] + NORM_SCALE[MARKER_INDEX['glucose']] * b
        ra_emb = M._RA_BASELINE_MAX_Z * torch.tanh(m.metabolic.ra_baseline_net(e_met).squeeze(-1))
        Ra = torch.nn.functional.softplus(m.metabolic.log_ra + ra_emb)
        sp = C._CVS_BASELINE_MAX_Z * torch.tanh(m.cardiovascular.setpoint_net(e_cvs))  # [K,4]
        cvs_idx = [MARKER_INDEX[x] for x in ('hr','hrv','sbp','dbp')]
        cvs = torch.stack([torch.tensor(NORM_CENTER[i]) + torch.tensor(NORM_SCALE[i]) * sp[:, j]
                           for j, i in enumerate(cvs_idx)], dim=1)
    return dict(Gb=Gb.numpy(), Ra=Ra.numpy(), HR0=cvs[:,0].numpy(), HRV0=cvs[:,1].numpy(),
                SBP0=cvs[:,2].numpy(), DBP0=cvs[:,3].numpy())

def summarize(tag, emb):
    p = physiology(emb)
    print(f'--- {tag} (n={emb.shape[0]}, mean ||emb||={emb.norm(dim=1).mean():.2f}) ---')
    for k, v in p.items():
        print(f'  {k:5}: min {v.min():7.2f}  p10 {np.percentile(v,10):7.2f}  med {np.median(v):7.2f}'
              f'  p90 {np.percentile(v,90):7.2f}  max {v.max():7.2f}  (std {v.std():6.2f})')
    return p

torch.manual_seed(0)
K = 4000
# 1) embeddings from the LEARNED prior (what a "typical" trained patient looks like)
prior = pm.unsqueeze(0) + ps.unsqueeze(0) * torch.randn(K, 64)
p_prior = summarize('PRIOR-sampled embeddings (the learned population)', prior)

# 2) embeddings the calibration LEASH allows (||emb||<=3.0) — the reachable set
wide = torch.randn(K, 64); wide = wide / wide.norm(dim=1, keepdim=True) * torch.rand(K,1)*3.0
summarize('LEASH-reachable embeddings (||emb||<=3.0, calibration search space)', pm.unsqueeze(0)+wide)

# 3) identifiability: cross-marker entanglement across prior-sampled patients.
print('\n=== cross-marker correlation over prior-sampled patients (entanglement) ===')
keys = list(p_prior.keys())
Mx = np.stack([p_prior[k] for k in keys])
Cmat = np.corrcoef(Mx)
print('       ' + ' '.join(f'{k:>6}' for k in keys))
for i,k in enumerate(keys):
    print(f'  {k:5} ' + ' '.join(f'{Cmat[i,j]:6.2f}' for j in range(len(keys))))

# 4) identifiability: does moving the embedding actually move each marker? (per-dim
#    max sensitivity of each physiological output to a unit embedding step)
print('\n=== identifiability: physiological span reachable within the leash ===')
print('    (a marker calibration CANNOT move has ~0 span => non-identifiable)')
for k in keys:
    v = physiology(pm.unsqueeze(0)+wide)[k]
    print(f'  {k:5}: reachable range [{v.min():.1f}, {v.max():.1f}]  span {v.max()-v.min():.1f}')
