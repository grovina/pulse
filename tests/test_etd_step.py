"""Exponential (ETD1) stepping of the linear part (PLAN B2).

``euler_step`` advances a restoring term −k·(x − x*) by (1 − k·dt) where the exact solution
contracts by e^{−k·dt}, so the time constant a module declares is not the one the rollout
integrates and the trajectory depends on dt. A module may report a per-marker decay rate
alongside its rate (``decay_rates``); the step then advances ``x + rate·h`` with
``h = (1 − e^{−k·dt})/k``. These tests pin, without a trained checkpoint:

* nothing changes until a module opts in — zero decay IS the head integrator, bit for bit;
* a pure relaxation lands on the analytic solution (where Euler is off by the percentages
  below), and the answer stops depending on dt;
* the step is finite and correctly differentiable at k = 0 and at large k;
* the special coordinates (HRV and pulse pressure in log space, SpO₂ in logit space, SBP rebuilt
  as DBP + PP) keep their invariants, with decay active.

Measured at dt = 1 min, on a pure relaxation (effective rate = −ln of the per-step factor):

    k      Euler factor   exact   Euler effective rate   error
    0.15      0.850       0.861         0.163             +8 %
    0.30      0.700       0.741         0.357            +19 %   cardiovascular default, teacher k_hr
    0.80      0.200       0.449         1.609           +101 %   cardiovascular band ceiling

``tests/test_planned_integrator.py`` pins the same channel through the planned rollout.
"""

from __future__ import annotations

import math

import pytest
import torch

import pulse.model as model_module
from pulse.model import (
    ModularPhysiologyNetwork,
    _etd_step_size,
    _reports_decay,
    euler_step,
    integrate,
    precompute_gut_outputs,
)
from pulse.modules.base import PhysiologyModule
from pulse.modules.cardiovascular import CardiovascularModule
from pulse.modules.gut import MealEvent
from pulse.modules.respiratory import RespiratoryModule
from pulse.modules.thermoreg import ThermoregModule
from pulse.types import EMBEDDING_DIM, MARKER_INDEX, NORM_CENTER, NORM_SCALE, STATE_DIM

HR, HRV, SBP, DBP, SPO2 = (MARKER_INDEX[m] for m in ("hr", "hrv", "sbp", "dbp", "spo2"))
GLUCOSE = MARKER_INDEX["glucose"]
SPECIAL = (HRV, SBP, DBP, SPO2)
PLAIN = [i for i in range(STATE_DIM) if i not in SPECIAL]


# ---- the integrator as it was before PLAN B2 (ea16d0d), verbatim ------------------------------

def _head_exp_step(x, rate, dt):
    positive = x > 0
    denom = torch.where(positive, x, torch.ones_like(x))
    return torch.where(positive, x * torch.exp(rate * dt / denom), x + rate * dt)


def _head_logit_interval_step(x, rate, dt, lo, hi):
    eps = 1e-4
    width = hi - lo
    x_c = x.clamp(lo + eps, hi - eps)
    u = (x_c - lo) / width
    logit = torch.log(u / (1.0 - u))
    dsdlogit = (x_c - lo) * (hi - x_c) / width
    logit_new = logit + rate * dt / dsdlogit.clamp(min=eps)
    return (lo + width * torch.sigmoid(logit_new)).clamp(lo + eps, hi - eps)


def head_euler_step(state, rates, dt):
    new_state = state + rates * dt
    hrv_new = _head_exp_step(state[..., HRV], rates[..., HRV], dt)
    pp = state[..., SBP] - state[..., DBP]
    pp_new = _head_exp_step(pp, rates[..., SBP] - rates[..., DBP], dt)
    dbp_new = new_state[..., DBP]
    spo2_new = _head_logit_interval_step(state[..., SPO2], rates[..., SPO2], dt, 70.0, 100.0)
    new_state = new_state.clone()
    new_state[..., HRV] = hrv_new
    new_state[..., SBP] = dbp_new + pp_new
    new_state[..., SPO2] = spo2_new
    return new_state


def _pair(batch: int | None, seed: int, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """A physiological random state (HRV and pulse pressure > 0, SpO₂ inside (70, 100)) and a
    random rate vector of the size the learned drivers produce."""
    g = torch.Generator().manual_seed(seed)
    shape = (STATE_DIM,) if batch is None else (batch, STATE_DIM)
    lead = shape[:-1]
    center = torch.tensor(NORM_CENTER, dtype=dtype)
    scale = torch.tensor(NORM_SCALE, dtype=dtype)
    state = center + 0.3 * torch.randn(shape, generator=g, dtype=dtype) * scale
    state[..., HRV] = 10.0 + 60.0 * torch.rand(lead, generator=g, dtype=dtype)
    state[..., DBP] = 60.0 + 30.0 * torch.rand(lead, generator=g, dtype=dtype)
    state[..., SBP] = state[..., DBP] + 20.0 + 40.0 * torch.rand(lead, generator=g, dtype=dtype)
    state[..., SPO2] = 90.0 + 9.0 * torch.rand(lead, generator=g, dtype=dtype)
    rates = 0.1 * torch.randn(shape, generator=g, dtype=dtype) * scale
    return state, rates


# ---- 1. nothing changes until a module opts in ------------------------------------------------

@pytest.mark.parametrize("dt", [1.0, 0.5, 2.5, 10.0])
@pytest.mark.parametrize("batch", [None, 1, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_decay_and_zero_decay_are_the_head_euler_step_bit_for_bit(dt, batch, dtype) -> None:
    state, rates = _pair(batch, 1, dtype)
    want = head_euler_step(state, rates, dt)
    assert torch.equal(euler_step(state, rates, dt), want)
    assert torch.equal(euler_step(state, rates, dt, None), want)
    assert torch.equal(euler_step(state, rates, dt, torch.zeros_like(rates)), want)
    # A decay broadcast from a single row (a parameter-only constant) is the same step.
    if batch:
        assert torch.equal(euler_step(state, rates, dt, torch.zeros(STATE_DIM, dtype=dtype)), want)


MEALS = [
    MealEvent(time=20.0, carbs=60.0, fats=20.0, proteins=25.0),
    MealEvent(time=-150.0, carbs=40.0, fats=10.0, proteins=15.0),
]


def _small_model(seed: int) -> ModularPhysiologyNetwork:
    """Small widths, every zero-init output layer woken so the embedding reaches the setpoints."""
    torch.manual_seed(seed)
    model = ModularPhysiologyNetwork(
        metabolic_hidden=12, appetite_hidden=8, stress_hidden=8,
        cardiovascular_hidden=12, thermoreg_hidden=8, respiratory_hidden=8,
        gut_hidden=8, hepatobiliary_hidden=8,
    )
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


def _rollout_inputs(batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return _pair(batch, seed + 1)[0], 0.4 * torch.randn(batch, EMBEDDING_DIM, generator=g)


def test_rollout_is_the_pre_b2_integrator_when_no_module_reports_decay(monkeypatch) -> None:
    """Through the whole planned machinery (and the pointwise reference): with no module
    reporting a decay the trajectory is the one the head integrator produced, to the last bit."""
    def head_integrator(state, rates, dt, decay=None):
        assert decay is None, "no module reports a decay, so none may reach the step"
        return head_euler_step(state, rates, dt)

    model = _small_model(11)
    state, emb = _rollout_inputs(3, 5)
    monkeypatch.setattr(model_module, "_reports_decay", lambda mod: False)
    for planned in (True, False):
        kw = dict(meals=MEALS, planned=planned, start_time_minutes=480.0)
        with torch.no_grad():
            now = integrate(model, state, emb, 40, **kw)
            monkeypatch.setattr(model_module, "euler_step", head_integrator)
            then = integrate(model, state, emb, 40, **kw)
            monkeypatch.setattr(model_module, "euler_step", euler_step)
        assert torch.equal(now, then), f"planned={planned}"


def test_modules_reporting_zero_decay_give_the_same_trajectory(monkeypatch) -> None:
    """A module that opts in with k = 0 changes nothing either: h = dt·1.0 exactly, so the
    ETD path through the plan (and through ``forward``) is the plain Euler trajectory."""
    model = _small_model(12)
    state, emb = _rollout_inputs(2, 6)
    gut = precompute_gut_outputs(model, emb, 40, meals=MEALS)
    kw = dict(meals=MEALS, gut_outputs=gut)
    reporting = (CardiovascularModule, ThermoregModule, RespiratoryModule)
    with torch.no_grad():
        monkeypatch.setattr(model_module, "_reports_decay", lambda mod: False)
        plain = {p: integrate(model, state, emb, 40, planned=p, **kw) for p in (True, False)}
        monkeypatch.undo()
        for cls in reporting:
            monkeypatch.setattr(
                cls, "decay_rates",
                lambda self, state, coupling, const, drv, raw: torch.zeros_like(state), raising=False)
        monkeypatch.setattr(model_module, "_reports_decay", lambda mod: type(mod) in reporting)
        zero = {p: integrate(model, state, emb, 40, planned=p, **kw) for p in (True, False)}
    for p in (True, False):
        assert torch.equal(zero[p], plain[p]), f"planned={p}"


def test_a_module_opts_in_by_defining_decay_rates() -> None:
    """The base class defines nothing (or a default meaning "none"); only an override counts."""
    m = ModularPhysiologyNetwork(
        metabolic_hidden=8, appetite_hidden=8, stress_hidden=8, cardiovascular_hidden=8,
        thermoreg_hidden=8, respiratory_hidden=8, gut_hidden=8, hepatobiliary_hidden=8)
    base = getattr(PhysiologyModule, "decay_rates", None)
    for name, mod in m._modules_by_name.items():
        own = getattr(type(mod), "decay_rates", None)
        assert _reports_decay(mod) == (own is not None and own is not base), name


# ---- 2. a pure relaxation lands on the analytic solution --------------------------------------

def _relaxation(k: float, dtype=torch.float32, k_pp: float | None = None):
    """Every marker relaxing to its NORM_CENTER from 0.35 NORM_SCALE away, rate = −k·dev, with
    HRV and pulse pressure relaxing in log space (rate = x·dlog x/dt). Pulse pressure's own
    rate constant ``k_pp`` (default k) lives in the ``sbp`` slot of the decay, DBP's in ``dbp``."""
    k_pp = k if k_pp is None else k_pp
    center = torch.tensor(NORM_CENTER, dtype=dtype)
    scale = torch.tensor(NORM_SCALE, dtype=dtype)
    state = center + 0.35 * scale
    state[SPO2] = 97.0
    rates = -k * (state - center)
    pp_star = center[SBP] - center[DBP]
    pp = state[SBP] - state[DBP]
    rates[HRV] = state[HRV] * (-k) * (torch.log(state[HRV]) - torch.log(center[HRV]))
    rates[SBP] = rates[DBP] + pp * (-k_pp) * (torch.log(pp) - torch.log(pp_star))
    decay = torch.full((STATE_DIM,), k, dtype=dtype)
    decay[SBP] = k_pp
    return state, rates, decay, center, pp_star


@pytest.mark.parametrize("k", [0.15, 0.3, 0.8])
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 5e-6)])
def test_pure_relaxation_matches_the_analytic_solution(k, dtype, tol) -> None:
    """One dt = 1 min step of dx/dt = −k·(x − x*) lands on x* + (x − x*)·e^{−k·dt}: to 1e-12 of
    the deviation in float64 (k = 0.3 and 0.8 take the closed form; 0.15 too), 5e-6 in float32.
    Euler contracts the deviation by 0.850 / 0.700 / 0.200 where the exact factor is 0.861 /
    0.741 / 0.449 — effective rates 0.163 (+8 %), 0.357 (+19 %) and 1.609 (+101 %), asserted in
    ``test_euler_misses_the_declared_time_constant_by_the_table`` below."""
    dt = 1.0
    k_pp = 1.5 * k  # unlike DBP's: the ``sbp`` slot of the decay is the pulse pressure's, not SBP's
    state, rates, decay, center, pp_star = _relaxation(k, dtype, k_pp)
    exact = math.exp(-k * dt)
    new = euler_step(state, rates, dt, decay)
    ulp = 4.0 * torch.finfo(dtype).eps * state.abs()
    # Ordinary markers: x* + (x − x*)·e^{−k·dt}.
    want = center + (state - center) * exact
    for i in PLAIN + [DBP]:
        assert abs(float(new[i] - want[i])) <= tol * abs(float(state[i] - center[i])) + float(ulp[i]), i
    # HRV and pulse pressure: the same relaxation on the LOG deviation.
    hrv_want = center[HRV] * (state[HRV] / center[HRV]) ** exact
    assert abs(float(new[HRV] / hrv_want - 1.0)) <= tol
    pp_want = pp_star * ((state[SBP] - state[DBP]) / pp_star) ** math.exp(-k_pp * dt)
    assert abs(float((new[SBP] - new[DBP]) / pp_want - 1.0)) <= 20 * tol
    assert float(new[SBP]) > float(new[DBP])


@pytest.mark.parametrize("k,euler_rate_error", [(0.15, 0.083), (0.3, 0.189), (0.8, 1.012)])
def test_euler_misses_the_declared_time_constant_by_the_table(k, euler_rate_error) -> None:
    """The motivation, measured: the rate Euler actually integrates a pure relaxation at is
    −ln(1 − k·dt), and the exponential step recovers k."""
    dt = 1.0
    state, rates, decay, center, _ = _relaxation(k, torch.float64)
    dev = float(state[GLUCOSE] - center[GLUCOSE])
    euler = head_euler_step(state, rates, dt)
    etd = euler_step(state, rates, dt, decay)
    realized_euler = -math.log(float(euler[GLUCOSE] - center[GLUCOSE]) / dev) / dt
    realized_etd = -math.log(float(etd[GLUCOSE] - center[GLUCOSE]) / dev) / dt
    assert realized_euler / k - 1.0 == pytest.approx(euler_rate_error, abs=1e-3)
    assert realized_etd == pytest.approx(k, rel=1e-12)


def test_a_driven_relaxation_is_exact_for_any_dt() -> None:
    """rate = F − k·(x − x*) with F held over the step: x_eq = x* + F/k, for any dt."""
    i = MARKER_INDEX["ffa"]
    k, force, x_star = 0.2, 0.05, 0.4
    state = torch.tensor(NORM_CENTER, dtype=torch.float64)
    state[i] = 1.3
    x_eq = x_star + force / k
    for dt in (1.0, 5.0, 30.0, 400.0):
        rates = torch.zeros(STATE_DIM, dtype=torch.float64)
        rates[i] = force - k * (state[i] - x_star)
        decay = torch.zeros(STATE_DIM, dtype=torch.float64)
        decay[i] = k
        new = euler_step(state, rates, dt, decay)
        assert float(new[i]) == pytest.approx(x_eq + (1.3 - x_eq) * math.exp(-k * dt), rel=1e-12)
        assert float(new[GLUCOSE]) == float(state[GLUCOSE])  # reports no decay, has no rate


def test_a_stiff_relaxation_never_overshoots() -> None:
    """Euler's factor is 1 − k·dt: at k = 50 it is −49 and the state explodes (glucose to
    −419 mg/dL). The exponential step goes to the target and stays between it and the state."""
    for k in (2.0, 50.0, 1e4):
        state, rates, decay, center, _ = _relaxation(k)
        new = euler_step(state, rates, 1.0, decay)
        for i in PLAIN:
            lo, hi = sorted((float(state[i]), float(center[i])))
            assert lo - 1e-4 * (hi - lo) <= float(new[i]) <= hi + 1e-4 * (hi - lo), i
        assert float(new[HRV]) > 0.0 and float(new[SBP]) > float(new[DBP])
        assert 70.0 < float(new[SPO2]) < 100.0
    state, rates, _, center, _ = _relaxation(50.0)
    blown = head_euler_step(state, rates, 1.0)
    assert float(blown[GLUCOSE] - center[GLUCOSE]) == pytest.approx(-49.0 * 10.5, rel=1e-5)


# ---- 3. the answer stops depending on dt ------------------------------------------------------

def _scalar_relaxation(i: int, k: float, x_star: float):
    def rates(s):
        r = torch.zeros(STATE_DIM, dtype=torch.float64)
        r[i] = -k * (s[i] - x_star)
        return r
    return rates


def _roll(state, rates_fn, decay, dt: float, n: int):
    s = state.clone()
    for _ in range(n):
        s = euler_step(s, rates_fn(s), dt, decay)
    return s


def test_dt_invariance_of_a_pure_relaxation() -> None:
    """The same 60 simulated minutes as 60 steps of dt = 1 and as 6 steps of dt = 10.

    k = 0.05, 40 units from the target (the largest k for which Euler is still stable at
    dt = 10, where its factor is 0.5). Remaining deviation after 60 min, exact 1.9915:

        Euler  dt=1  1.8428 (−7.5 %)   dt=10  0.6250 (−68.6 %)   the runs differ by 3.04 % of
                                                                  the initial deviation
        ETD    dt=1  1.9915 (< 1e-8)   dt=10  1.9915 (< 1e-11)   they differ by 7e-11 of it
    """
    k, x0, x_star, T = 0.05, 135.0, 95.0, 60
    base = torch.tensor(NORM_CENTER, dtype=torch.float64)
    base[GLUCOSE] = x0
    rates = _scalar_relaxation(GLUCOSE, k, x_star)
    decay = torch.zeros(STATE_DIM, dtype=torch.float64)
    decay[GLUCOSE] = k
    exact = (x0 - x_star) * math.exp(-k * T)

    euler_fine = float(_roll(base, rates, None, 1.0, 60)[GLUCOSE]) - x_star
    euler_coarse = float(_roll(base, rates, None, 10.0, 6)[GLUCOSE]) - x_star
    etd_fine = float(_roll(base, rates, decay, 1.0, 60)[GLUCOSE]) - x_star
    etd_coarse = float(_roll(base, rates, decay, 10.0, 6)[GLUCOSE]) - x_star
    assert euler_fine == pytest.approx(40.0 * 0.95 ** 60, rel=1e-9)
    assert euler_coarse == pytest.approx(40.0 * 0.5 ** 6, rel=1e-9)
    # dt = 1 is k·dt = 0.05, inside the series (truncation z⁵/720 per step: 1.4e-9 over 60 steps).
    assert etd_fine == pytest.approx(exact, rel=1e-8)
    assert etd_coarse == pytest.approx(exact, rel=1e-12)
    euler_gap = abs(euler_fine - euler_coarse) / (x0 - x_star)
    etd_gap = abs(etd_fine - etd_coarse) / (x0 - x_star)
    assert euler_gap == pytest.approx(0.0304, abs=5e-4)
    assert etd_gap < 1e-6 < euler_gap

    # At k = 0.3 Euler is not merely wrong at dt = 10 but unstable (|1 − 3| = 2 per step: 16-fold
    # growth of the deviation over four steps); the exponential step is still on the exact answer.
    k = 0.3
    rates = _scalar_relaxation(GLUCOSE, k, x_star)
    decay[GLUCOSE] = k
    diverged = float(_roll(base, rates, None, 10.0, 4)[GLUCOSE]) - x_star
    settled = float(_roll(base, rates, decay, 10.0, 4)[GLUCOSE]) - x_star
    assert diverged == pytest.approx(40.0 * 2.0 ** 4, rel=1e-9)
    assert settled == pytest.approx(40.0 * math.exp(-k * 40.0), rel=1e-6)


def test_dt_invariance_in_the_log_and_logit_coordinates() -> None:
    """HRV (log space) is as exact as an ordinary marker. SpO₂ (logit space) agrees with itself
    across dt far better than Euler does: k = 0.22, 94 → 98 %, 60 min as 6 steps of 10 ends
    3.04 points from the 60-step run under Euler and within 1e-3 under ETD."""
    base = torch.tensor(NORM_CENTER, dtype=torch.float64)
    base[HRV], base[SPO2] = 70.0, 94.0
    hrv_star, spo2_star, k = 40.0, 98.0, 0.22

    def rates(s):
        r = torch.zeros(STATE_DIM, dtype=torch.float64)
        r[HRV] = s[HRV] * (-k) * (torch.log(s[HRV]) - math.log(hrv_star))
        r[SPO2] = -k * (s[SPO2] - spo2_star)
        return r

    decay = torch.zeros(STATE_DIM, dtype=torch.float64)
    decay[HRV] = decay[SPO2] = k
    fine = _roll(base, rates, decay, 1.0, 60)
    coarse = _roll(base, rates, decay, 10.0, 6)
    want_hrv = hrv_star * (70.0 / hrv_star) ** math.exp(-k * 60.0)
    assert float(fine[HRV]) == pytest.approx(want_hrv, rel=1e-12)
    assert float(coarse[HRV]) == pytest.approx(want_hrv, rel=1e-12)
    assert abs(float(fine[SPO2] - coarse[SPO2])) < 1e-3
    euler_gap = abs(float(_roll(base, rates, None, 1.0, 60)[SPO2] - _roll(base, rates, None, 10.0, 6)[SPO2]))
    assert euler_gap == pytest.approx(3.04, abs=0.05)


# ---- 4. the special coordinates keep their invariants -----------------------------------------

def test_spo2_stays_in_its_interval_for_any_decay_and_driver() -> None:
    state = torch.tensor(NORM_CENTER, dtype=torch.float32)
    for k in (0.0, 0.08, 0.22, 5.0, 1e4):
        for drive in (80.0, -80.0, 1e4):
            s = state.clone()
            decay = torch.zeros(STATE_DIM)
            decay[SPO2] = k
            for _ in range(60):
                rates = torch.zeros(STATE_DIM)
                rates[SPO2] = drive - k * (s[SPO2] - 98.0)
                s = euler_step(s, rates, 1.0, decay)
            assert 70.0 < float(s[SPO2]) < 100.0, (k, drive)


def test_spo2_decay_vanishes_into_the_logit_euler_step_and_tracks_the_exponential() -> None:
    """k → 0 is the Euler logit step; at a small deviation the exponential step lands on the
    SpO₂-space exponential to the logit map's curvature (1.8e-3 points here: 97.5 → 98, k = 0.22)."""
    state = torch.tensor(NORM_CENTER, dtype=torch.float64)
    state[SPO2] = 97.5
    rates = torch.zeros(STATE_DIM, dtype=torch.float64)
    rates[SPO2] = -0.22 * (97.5 - 98.0)
    near_zero = torch.zeros(STATE_DIM, dtype=torch.float64)
    near_zero[SPO2] = 1e-9
    assert float(euler_step(state, rates, 1.0, near_zero)[SPO2]) == pytest.approx(
        float(euler_step(state, rates, 1.0)[SPO2]), abs=1e-8)
    decay = torch.zeros(STATE_DIM, dtype=torch.float64)
    decay[SPO2] = 0.22
    exact = 98.0 + (97.5 - 98.0) * math.exp(-0.22)
    got = float(euler_step(state, rates, 1.0, decay)[SPO2])
    euler = float(euler_step(state, rates, 1.0)[SPO2])
    assert got == pytest.approx(exact, abs=3e-3)
    assert abs(got - exact) < abs(euler - exact)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sbp_above_dbp_and_hrv_positive_under_hostile_drivers_and_any_decay(dtype) -> None:
    """The cardiovascular module's structure — log-space HRV and pulse pressure, each with a
    restoring term k = 0.8 — under a driver of −5 /min (a 150-fold collapse per minute that
    additive Euler would carry through zero on step 1), with decays from none to stiff that do
    NOT match the rate's k: the log-space step cannot cross zero and SBP is DBP + PP."""
    batch = 16
    state, noise = _pair(batch, 3, dtype)
    g = torch.Generator().manual_seed(9)
    s = state.clone()
    for step in range(300):
        r = 0.2 * noise * torch.randn(noise.shape, generator=g, dtype=dtype)
        hrv, pp = s[..., HRV], s[..., SBP] - s[..., DBP]
        r[..., HRV] = hrv * (-5.0 - 0.8 * (torch.log(hrv) - math.log(40.0)))
        r[..., DBP] = -1.0 - 0.8 * (s[..., DBP] - 80.0)
        r[..., SBP] = r[..., DBP] + pp * (-5.0 - 0.8 * (torch.log(pp) - math.log(40.0)))
        decay = torch.rand(noise.shape, generator=g, dtype=dtype) ** 3 * 5.0
        decay[:, ::3] = 0.0
        s = euler_step(s, r, 1.0, decay)
        assert bool((s[..., HRV] > 0).all()), step
        assert bool((s[..., SBP] > s[..., DBP]).all()), step
        assert bool(((s[..., SPO2] > 70) & (s[..., SPO2] < 100)).all()), step
    assert bool(torch.isfinite(s).all())


def test_invariants_hold_through_integrate_for_random_embeddings(monkeypatch) -> None:
    """SBP > DBP and HRV > 0 for random embeddings at the calibration leash (‖e‖ = 3) with the
    cardiovascular decays at the band's ceiling (0.8, so Euler would be at factor 0.2), also with
    a hostile driver on HRV and pulse pressure — and the decay really changed the trajectory."""
    def decay_of(k: float):
        return lambda self, state, coupling, const, drv, raw: torch.full_like(state, k)

    B, T = 6, 240
    g = torch.Generator().manual_seed(21)
    emb = torch.randn(B, EMBEDDING_DIM, generator=g)
    emb = 3.0 * emb / emb.norm(dim=-1, keepdim=True)
    state0 = torch.tensor(NORM_CENTER, dtype=torch.float32).expand(B, -1)
    for hostile in (False, True):
        model = _small_model(5)
        if hostile:
            with torch.no_grad():
                model.cardiovascular.network[-1].bias[1] = -50.0
                model.cardiovascular.network[-1].bias[2] = -80.0
        reporting = (CardiovascularModule, ThermoregModule, RespiratoryModule)
        with torch.no_grad():
            monkeypatch.setattr(model_module, "_reports_decay", lambda mod: False)
            euler = integrate(model, state0, emb, T)
            monkeypatch.undo()
            for cls, k in zip(reporting, (0.8, 0.1, 0.2)):
                monkeypatch.setattr(cls, "decay_rates", decay_of(k), raising=False)
            monkeypatch.setattr(model_module, "_reports_decay", lambda mod: type(mod) in reporting)
            etd = integrate(model, state0, emb, T)
            monkeypatch.undo()
        for name, tr in (("euler", euler), ("etd", etd)):
            assert bool(torch.isfinite(tr).all()), (hostile, name)
            assert bool((tr[..., SBP] > tr[..., DBP]).all()), (hostile, name)
            assert bool((tr[..., HRV] > 0).all()), (hostile, name)
            assert bool(((tr[..., SPO2] > 70) & (tr[..., SPO2] < 100)).all()), (hostile, name)
        assert float((etd - euler).abs().max()) > 1e-3, "the decay channel was not exercised"


# ---- 5. differentiable at k = 0 and at large k ------------------------------------------------

def _naive_step_size(k: torch.Tensor, dt: float) -> torch.Tensor:
    """The textbook guard: ``dt`` where k == 0, else the closed form. Its forward pass is right."""
    return torch.where(k == 0, torch.full_like(k, dt), -torch.expm1(-k * dt) / k)


def test_the_naive_guard_has_a_nan_gradient_at_zero_and_ours_does_not() -> None:
    k = torch.zeros(4, requires_grad=True)
    _naive_step_size(k, 1.0).sum().backward()
    assert bool(torch.isnan(k.grad).all()), "the trap: where() backpropagates through the dead branch"
    k = torch.zeros(4, requires_grad=True)
    _etd_step_size(k, 1.0).sum().backward()
    assert bool(torch.isfinite(k.grad).all())


@pytest.mark.parametrize("dt", [0.5, 1.0, 10.0])
def test_gradient_at_exactly_zero_decay_is_finite_and_correct(dt) -> None:
    """d(step)/dk = −dt²/2 at k = 0, so d(x_new)/dk = −rate·dt²/2 for an ordinary marker (and
    out·rate/x·(−dt²/2) for the log-space one). A clamp(min=0) on k would make this exactly 0."""
    state, rates = _pair(5, 4, torch.float64)
    state, rates = state.requires_grad_(), rates.requires_grad_()
    decay = torch.zeros_like(rates, requires_grad=True)
    out = euler_step(state, rates, dt, decay)
    w = torch.linspace(0.5, 1.5, STATE_DIM, dtype=torch.float64)
    (out * w).sum().backward()
    for g in (state.grad, rates.grad, decay.grad):
        assert bool(torch.isfinite(g).all())
    for i in PLAIN:
        assert torch.allclose(decay.grad[:, i], w[i] * rates[:, i].detach() * (-dt * dt / 2), rtol=1e-12), i
        # and d(x_new)/d(rate) is the effective step, which is dt at k = 0
        assert torch.allclose(rates.grad[:, i], torch.full((5,), dt, dtype=torch.float64) * w[i]), i
    hrv_state = state.detach()[:, HRV]
    want = w[HRV] * out[:, HRV].detach() * rates.detach()[:, HRV] / hrv_state * (-dt * dt / 2)
    assert torch.allclose(decay.grad[:, HRV], want, rtol=1e-10)


def test_gradcheck_through_the_whole_step() -> None:
    """Central differences agree with autograd, with decays at zero, tiny, inside the series,
    either side of its edge, and stiff, in every coordinate."""
    state, rates = _pair(3, 5, torch.float64)
    ladder = [1e-9, 1e-4, 0.01, 0.05, 0.09, 0.11, 0.15, 0.3, 0.8, 2.0, 7.0, 40.0]
    decay = torch.zeros_like(rates)
    decay[1] = torch.tensor((ladder * 3)[:STATE_DIM], dtype=torch.float64)
    decay[2] = 0.3
    for t in (state, rates, decay):
        t.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda s, r, d: euler_step(s, r, 1.0, d), (state, rates, decay), eps=1e-6, atol=1e-6, rtol=1e-6)


def test_step_size_matches_float64_in_value_and_gradient() -> None:
    """The series/closed-form switch at |k·dt| = 0.1 is invisible in float32: value within 5e-7
    and derivative within 1e-5 of the float64 formula from k = 0 to k = 1e3 (the closed form
    alone is off by up to 300 % in the derivative below z = 1e-6)."""
    grid = torch.cat([
        torch.zeros(1), torch.logspace(-9, 3, 2000, dtype=torch.float64).float(),
        torch.tensor([0.0999, 0.0999999, 0.1, 0.1000001, 0.1001]),
    ])
    for dt in (1.0, 10.0):
        k = grid.clone().requires_grad_()
        h = _etd_step_size(k, dt)
        (g,) = torch.autograd.grad(h.sum(), k)
        k64 = grid.double().requires_grad_()
        z = k64 * dt
        tiny = z < 1e-12
        h64 = dt * torch.where(tiny, 1 - z / 2, -torch.expm1(-z) / torch.where(tiny, torch.ones_like(z), z))
        (g64,) = torch.autograd.grad(h64.sum(), k64)
        assert bool(torch.isfinite(h).all() and torch.isfinite(g).all())
        assert float(((h.detach().double() - h64.detach()).abs() / h64.detach()).max()) < 5e-7
        assert float(((g.double() - g64).abs() / g64.abs()).max()) < 1e-5


@pytest.mark.parametrize("k", [1e3, 1e6, 1e12, 1e30])
def test_gradient_is_finite_at_large_decay(k) -> None:
    for dtype in (torch.float32, torch.float64):
        state, rates = _pair(None, 6, dtype)
        state, rates = state.requires_grad_(), rates.requires_grad_()
        decay = torch.full((STATE_DIM,), k, dtype=dtype, requires_grad=True)
        out = euler_step(state, rates, 1.0, decay)
        out.sum().backward()
        assert bool(torch.isfinite(out).all()), dtype
        for g in (state.grad, rates.grad, decay.grad):
            assert bool(torch.isfinite(g).all()), dtype
    # h → 1/k, so a plain marker moves by rate/k and d(step)/dk = −1/k² → 0.
    kk = torch.full((1,), k, dtype=torch.float64, requires_grad=True)
    h = _etd_step_size(kk, 1.0)
    h.backward()
    assert float(h.detach()) == pytest.approx(1.0 / k, rel=1e-12)
    assert float(kk.grad) == pytest.approx(-1.0 / k ** 2, rel=1e-9)


def test_mixed_zero_and_nonzero_decay_gives_no_nan_gradient_anywhere() -> None:
    """The usual case: most markers report nothing (exactly 0) next to a few that do."""
    state, rates = _pair(6, 7)
    state, rates = state.requires_grad_(), rates.requires_grad_()
    decay = torch.zeros(6, STATE_DIM)
    decay[:, [HR, HRV, SBP, DBP]] = 0.3
    decay[:, SPO2] = 0.08
    decay.requires_grad_()
    euler_step(state, rates, 1.0, decay).pow(2).sum().backward()
    for g in (state.grad, rates.grad, decay.grad):
        assert bool(torch.isfinite(g).all())
    assert float(decay.grad[:, GLUCOSE].abs().max()) > 0.0  # d/dk at k = 0 is live
