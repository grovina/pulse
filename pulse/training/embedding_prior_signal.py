"""
Embedding-prior signal — shape the patient code so the eval prior is well-specified.

Iter 91. Calibration finds a new person by SOLVING for their embedding from their observations.
That is an inverse problem, and it is underdetermined: a calibration window supplies ~5 markers
x ~10 check-ins ~= 50 numbers, against (before this iter) 64 unknowns. Measured consequence --
the recovered embedding is essentially ORTHOGONAL to the truth (cos ~= 0) while still explaining
the data, and fasting-glucose recovery is worse than simply predicting the population mean.

For an underdetermined inverse problem the correct treatment is a PRIOR. iter 91 re-enables the
calibration prior (benchmark.py, PRIOR_WEIGHT 0 -> 1.0; measured: Gb skill -1.47 -> +0.21) and
halves the number of unknowns (EMBEDDING_DIM 64 -> 32, types.py).

But that prior is a DIAGONAL GAUSSIAN N(prior_mean, prior_std^2) fitted POST-HOC to the trained
embedding table -- and nothing during training ever shaped that table. It is whatever cloud the
reconstruction loss happened to leave behind: measured on iter-90, ||prior_mean|| = 0.26 (a
well-formed code would be centred at 0) with per-dimension scales differing 1.7x. So the eval
prior is an approximation of an arbitrary cloud, when it could be exact.

Two terms close that gap and they do different jobs.

SCALE (``weight``, spec 0.001): mean_i ||e_i||^2. Compacts and isotropises the cloud, so
N(0, sigma^2 I) is the distribution the codes were trained to have rather than a post-hoc fit. It
also counteracts the failure mode the norm clamp was bolted on to contain (iter 81): calibration
walking the embedding far off the trained manifold, where the ODE detonates. Deliberately weak. It
is a regulariser, not an objective: it must shape the cloud without flattening the per-patient
information the setpoint and meal-response signals put there. If it is too strong the embeddings
collapse toward zero and every patient looks like the population mean -- which would show up
immediately as a rise in setpoint_supervision's per-marker MAE.

CENTRE (``center_weight``, default 1.0; plan A2): ||mean_i e_i||^2. The scale term does not centre
the cloud. It contains the centre (mean||e_i||^2 = ||m||^2 + mean||e_i - m||^2, m = mean_i e_i),
but at 0.001 that is a pull of 2*w*||m||/sqrt(N) = 8e-5 in table-gradient norm at ||m|| = 0.26 (the
iter-90 offset) and N = 40, against reconstruction gradients clipped at 10. Nothing else holds it:
every consumer of a code reaches it through an affine map (the per-module ``embedding_projections``
and ``default_inputs_net``, all with biases), so shifting every row by d and each bias by -W d
leaves every patient's physiology exactly unchanged. The centre is a gauge direction for the rows --
the reconstruction loss is blind to it and the cloud sits wherever it drifted. Only the zero
embedding, which the table does not move, feels the offset.

That is how one objective came to carry two centres. The calibration prior (benchmark.py,
calibration.py) is built at mean(table); the soft norm penalty (radius 3), the hard clamp
(||e|| <= 8), the zero-embedding supervision (default patients, the sweeps) and the
textbook/gate evaluation point are all centred at ZERO. Pinning mean(E) to 0 makes them one point,
and that is the sense in which N(0, diag sigma) is the distribution the codes were TRAINED to have
rather than one FITTED to them afterwards: ``embedding_prior_mean`` stops being an estimate of where
the cloud landed and becomes the fixed point of a constraint (~0), with ``embedding_prior_std`` the
scale that the reconstruction and setpoint losses chose around the origin the rest of the system is
already anchored at.

Centred is only the right thing because setpoints decode in log space (plan A1). The teacher draws
them lognormally, so a cloud centred on zero decodes, at its centre, to the MEDIAN person
(PatientParams(), the zero embedding) and the mean of the decoded population sits above it by
Jensen, E[exp x] > exp E[x] (Si mean/median 1.14, Ib 1.08, HRV0 1.09). With additive
``center + scale*tanh(head)`` decoders the same cloud gives E[setpoint] = center, which cannot be
right at the median and in the mean at once, so pinning the centre would have meant choosing
between them.

Strong is safe here where it is not for the scale term. The gradient of ||m||^2 is 2m/N on EVERY
row, so under the loss alone it moves the cloud rigidly and the pairwise differences e_i - e_j --
the per-patient information -- are untouched. It constrains ONE 32-vector rather than 40 of them,
which is why it wants a weight three orders above the per-row one: a constraint, not a nudge. The
two weights are independent; either can be zero and the signal still runs on the other. The scale
term also carries w*||m||^2 of centring, negligible next to ``center_weight``.

How strong "strong" must be is set by the optimizer, not by the loss scale. The pin lands only at
joint aux steps (spec: every 12 windows, 6 per epoch), while Adam divides each coordinate's step by
the RMS (sigma) of every gradient that coordinate sees, trajectory windows included. One pin step
then moves a coordinate by lr*g/sigma, and the centre relaxes with a time constant of
sigma*N/(2*lr*cw) pin steps. Toy at the spec's cadence and lr schedule (40 rows, an assumed 0.3
per-coordinate window gradient, ||m|| starting at 0.26): ||m|| at the end of phase 1 is 0.15 at
cw=1, 0.03 at cw=10, 0.01 at cw=30. So 1.0 is a constraint when the pin lands on every optimizer
step and a nudge at 6 per epoch; the recipe, which owns the cadence, sets the weight. The
``centre_norm`` on the trainer's per-epoch ``[EMBPRIOR]`` line is the measurement the toy stands
in for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .safe_step import accumulate_grad
from .signals import SignalContext, SignalResult, TrainingSignal, WeightSchedule

# Three orders above the 0.001 per-row weight. The shared default for the dataclass, ``train()`` and
# ``--embedding-prior-center-weight``, so the three cannot disagree (test_argparse_defaults_...).
DEFAULT_CENTER_WEIGHT = 1.0


@dataclass
class EmbeddingPriorSignal(TrainingSignal):
    """L2 on the patient codes: per row (``weight``) and on their mean (``center_weight``)."""

    weight: WeightSchedule = field(default_factory=lambda: WeightSchedule(0.0))
    center_weight: WeightSchedule = field(
        default_factory=lambda: WeightSchedule(DEFAULT_CENTER_WEIGHT)
    )

    name: str = "embedding_prior"
    source: str = "Iter 91: calibration is an underdetermined inverse problem; give it a prior"
    category: str = "mechanism"

    def weight_at(self, epoch: int) -> float:
        return self.weight.at(epoch)

    def center_weight_at(self, epoch: int) -> float:
        return self.center_weight.at(epoch)

    def compute(
        self,
        model: nn.Module,
        embeddings: nn.Embedding,
        ctx: SignalContext,
    ) -> SignalResult:
        w = self.weight_at(ctx.epoch)
        cw = self.center_weight_at(ctx.epoch)
        if w <= 0 and cw <= 0:
            return SignalResult()

        emb = embeddings.weight                       # [N, EMB] — every patient code
        centre = emb.mean(dim=0)                      # [EMB]
        loss = emb.pow(2).sum(dim=-1).mean()          # mean squared norm
        centre_loss = centre.pow(2).sum()             # squared norm of the mean

        with torch.no_grad():
            norms = emb.norm(dim=-1)
            sub = {
                "emb_norm_mean": float(norms.mean()),
                "emb_norm_max": float(norms.max()),
                # The eval prior is a diagonal Gaussian; it is only exact if the cloud is
                # centred. Watch this go toward 0 -- under center_weight, not weight.
                "emb_centre_norm": float(centre.norm()),
                # ...and isotropic. Watch this go toward 1.
                "emb_std_ratio": float(
                    emb.std(dim=0).max() / emb.std(dim=0).min().clamp_min(1e-6)
                ),
                # The per-patient information the centre term must leave alone: it should hold
                # while emb_centre_norm falls. If it falls too, the weight is flattening patients.
                "emb_spread": float((emb - centre).norm(dim=-1).mean()),
            }

        accumulate_grad(
            w * loss + cw * centre_loss, ctx, signal=self.name,
            extra={
                "raw_loss": float(loss.detach().item()), "weight": float(w),
                "raw_centre_loss": float(centre_loss.detach().item()), "center_weight": float(cw),
                **sub,
            },
        )
        return SignalResult(loss_sum=float(loss.detach().item()), n_units=1, sub_metrics=sub)
