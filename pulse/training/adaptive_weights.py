"""
Violation-proportional reweighting between the members of one signal
(physiology rules, cohort specs) — iter 97 form.

Iters 67/74 introduced "adaptive" weighting: a per-member EMA of the
violation, and per-member weights ``budget * ema_i / sum(ema)``. Two things
were wrong with it (review 4.3 / 4.6):

* it REPLACED the hand-set ``weight`` of each member, so a spec the author
  had deliberately down-weighted (``ffa_inverse_to_insulin`` at 0.5,
  ``sleep_hr_dip`` at 3x) was re-weighted purely by how badly it was missed;
* nothing bounded one member's share — measured on the iter-95 CCK case a
  single spec took 98.5 % of the cohort budget, and three timing rules took
  52 % of the rules budget while 47 rules sat at the floor.

``adaptive_multipliers`` keeps the intent (badly-missed members pull harder,
satisfied ones fade, the signal's total pull is preserved) and fixes both: the
result is ``base_i * m_i`` with ``sum(base_i * m_i) == sum(base_i)``, and no
member's share of the total exceeds ``cap_share``. The EMA must be fed a
DIMENSIONLESS violation (``violation / scale`` for rules, ``z^2`` for cohort
specs), otherwise a rule measured in minutes outranks one in mmol/L by 10^3
before any physiology is consulted.
"""

from __future__ import annotations


def adaptive_multipliers(
    ema: dict[str, float],
    base_weights: dict[str, float],
    *,
    cap_share: float = 0.25,
    floor_frac: float = 0.02,
    max_iter: int = 20,
) -> dict[str, float]:
    """Per-member weights ``base_i * m_i`` from a violation EMA.

    ``ema`` maps member name -> current (dimensionless) violation EMA; members
    missing from it keep their base weight. ``floor_frac`` floors every EMA at
    that fraction of the largest so a satisfied member keeps a residual pull
    (and can recover if drift later re-violates it). ``cap_share`` bounds any
    member's share of ``sum(base_weights)``; excess is redistributed
    proportionally over the uncapped members. The sum of the returned weights
    equals ``sum(base_weights)`` (up to float rounding).
    """
    names = list(base_weights)
    if not names:
        return {}
    budget = float(sum(base_weights.values()))
    if budget <= 0.0:
        return dict(base_weights)
    known = {k: float(v) for k, v in ema.items() if k in base_weights}
    if not known:
        return dict(base_weights)
    max_ema = max(known.values())
    if max_ema <= 0.0:
        return dict(base_weights)
    floor = floor_frac * max_ema
    # Members without an EMA yet behave as if they sat at the population mean
    # (multiplier 1) rather than at zero.
    mean_ema = sum(max(v, floor) for v in known.values()) / len(known)
    mult = {
        k: (max(known[k], floor) / mean_ema) if k in known else 1.0
        for k in names
    }
    raw = {k: base_weights[k] * mult[k] for k in names}
    total = sum(raw.values()) or 1e-12
    w = {k: budget * v / total for k, v in raw.items()}

    # Cap any member's share and redistribute the excess over the uncapped ones.
    n = len(names)
    # A cap below 1/n is infeasible; raising it TO 1/n would pin every member
    # to the same weight (two specs at 0.5 each), which defeats adapting. So the
    # cap applies only when the member count can honour it (n >= 1/cap_share,
    # i.e. 4 at the default 0.25 — the registries have 20-61); below that the
    # budget is uncapped and only the EMA ordering moves it.
    cap = (cap_share if cap_share * n >= 1.0 - 1e-9 else 1.0) * budget
    capped: set[str] = set()
    for _ in range(max_iter):
        over = {k for k in names if k not in capped and w[k] > cap * (1.0 + 1e-9)}
        if not over:
            break
        excess = sum(w[k] - cap for k in over)
        for k in over:
            w[k] = cap
        capped |= over
        free = [k for k in names if k not in capped]
        if not free:
            break
        free_total = sum(w[k] for k in free) or 1e-12
        for k in free:
            w[k] += excess * w[k] / free_total
    return w


__all__ = ["adaptive_multipliers"]
