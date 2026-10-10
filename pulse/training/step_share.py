"""How much of a training epoch each family of evidence actually received.

PLAN M1.8. Iter 97's literature stack took 25 optimizer contributions in a
21-hour run and pointwise teacher imitation took the rest. A movement is not
finished while that is still the shape of the log.

A contribution is one recorded pre-clip gradient norm. Trajectory norms are
tape steps. Each auxiliary list entry is one time that signal joined an
auxiliary step; a joint step that carries three literature signals counts
three literature contributions. Norm share is the sum of those norms over
the sum across every family.
"""

from __future__ import annotations

LITERATURE = frozenset({"cohort_statistic", "physiology_rules", "dose_response"})
TAPE = frozenset({"trajectory_rollout"})
DISTILL = frozenset({"cold_model_distillation"})
FAMILIES = ("literature", "tape", "distill", "other")


def family_of(name: str) -> str:
    if name in LITERATURE:
        return "literature"
    if name in TAPE:
        return "tape"
    if name in DISTILL:
        return "distill"
    return "other"


def step_share(
    *,
    aux_grad_norms: dict[str, list[float]],
    traj_grad_norms: list[float],
) -> dict[str, dict[str, float]]:
    steps = {fam: 0.0 for fam in FAMILIES}
    mass = {fam: 0.0 for fam in FAMILIES}
    for name, norms in aux_grad_norms.items():
        fam = family_of(name)
        steps[fam] += float(len(norms))
        mass[fam] += float(sum(norms))
    steps["tape"] += float(len(traj_grad_norms))
    mass["tape"] += float(sum(traj_grad_norms))
    total = sum(mass.values())
    return {
        fam: {
            "steps": steps[fam],
            "norm_mass": mass[fam],
            "norm_share": (mass[fam] / total) if total > 0.0 else 0.0,
        }
        for fam in FAMILIES
    }
