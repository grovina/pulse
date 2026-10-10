"""The literature queue.

Two lists live in one registry. A paper is filed as evidence items. An item
is ready only when the marker exists, the pathway can carry a gradient, and
the protocol we can run is the protocol the paper ran. Everything else is
blocked on a hub from PLAN.md, or sitting in the backlog because the number
belongs to a different population or a different protocol.

An agent integrates a ready item by landing it on the surface named here
(cohort spec, physiology rule, coupling prior) and setting ``status`` to
integrated with ``spec_name`` pointing at the registry entry. When that is
done, the agent reads another paper and files it. A blocked paper stays
blocked. The ready list is allowed to be empty.

The integrated corpus already in ``cohort_statistics``, ``physiology_rules``,
and ``coupling_priors`` is not copied here. This registry is the work that
is not yet that corpus, plus the items landed from this queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Status(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"
    BACKLOG = "backlog"
    INTEGRATED = "integrated"


# Hubs from PLAN.md. A blocked item names one of these.
HUBS: frozenset[str] = frozenset({
    "m1.4",  # second-phase insulin
    "m1.5",  # priors on the frozen constants
    "m2.1",  # sympathoadrenal
    "m2.2",  # stomach
    "m2.3",  # clock and sleep debt
    "m2.5",  # incretin and islet
    "m2.6",  # bile links out
    "m2.7",  # renal glucose
    "m2.8",  # stress input and drugs
    "m2.9",  # slow adaptation
})


@dataclass(frozen=True)
class QueueItem:
    id: str
    citation: str
    status: Status
    surface: str
    claim: str
    marker_id: str | None = None
    blocked_on: str | None = None
    spec_name: str | None = None
    note: str = ""


QUEUE: tuple[QueueItem, ...] = (
    QueueItem(
        id="dbp-aerobic-ceiling",
        citation="Lim et al. (1996); ACSM exercise testing guidelines",
        status=Status.INTEGRATED,
        surface="rule",
        claim=(
            "During moderate aerobic exercise diastolic pressure changes "
            "little or falls. Ceiling: a rise of 10 mmHg or more on the "
            "moderate-bout arm is a violation; a fall is free."
        ),
        marker_id="dbp",
        spec_name="dbp_does_not_rise_with_exercise",
    ),
    QueueItem(
        id="ffa-rises-in-exercise",
        citation="Frayn, Metabolic Regulation — exercise lipolysis",
        status=Status.INTEGRATED,
        surface="rule",
        claim="Free fatty acids rise during a moderate bout.",
        marker_id="ffa",
        spec_name="ffa_rises_with_exercise",
        note=(
            "The rule is already in the registry. The gradient is dead until "
            "m2.1 gives lipolysis an epinephrine drive. Do not add a second spec."
        ),
    ),
    QueueItem(
        id="cryer-1987-hierarchy",
        citation="Cryer et al. (1987) J Clin Invest 112884",
        status=Status.BLOCKED,
        surface="rule",
        claim=(
            "Glycaemic thresholds, in order: epinephrine 69 ± 2 mg/dL, "
            "glucagon 68 ± 2, growth hormone 66 ± 2, cortisol 58 ± 3, "
            "symptoms 53 ± 2."
        ),
        marker_id="glucose",
        blocked_on="m2.1",
        note="Both models fire hypoglycaemia into CRH at 70 mg/dL. There is no epinephrine state.",
    ),
    QueueItem(
        id="donga-2010-sleep-clamp",
        citation="Donga et al. (2010) J Clin Endocrinol Metab — one night of 4 h sleep",
        status=Status.BLOCKED,
        surface="cohort",
        claim=(
            "One night of 4 h versus 8 h sleep: peripheral insulin sensitivity "
            "−19 to −25% (clamp). Hepatic sensitivity essentially unchanged."
        ),
        marker_id="glucose",
        blocked_on="m2.3",
        note=(
            "sleep_restriction_next_day_glucose already targets +6 mg/dL and "
            "the current teacher moves glucose by <0.05%. The finding is a "
            "peripheral-sensitivity change, which needs sleep debt and the "
            "muscle-versus-liver split to act."
        ),
    ),
    QueueItem(
        id="wahren-1971-exercise-hgo",
        citation="Wahren et al. (1971) J Clin Invest — splanchnic glucose output in exercise",
        status=Status.BLOCKED,
        surface="cohort",
        claim=(
            "Healthy postabsorptive men, upright bicycle, 40 min. Splanchnic "
            "glucose production (estimated hepatic blood flow × arterial-hepatic "
            "venous glucose difference), mmoles/min, mean ± SE. Rest: "
            "0.80 ± 0.06 at 400 kg-m/min (n = 5), 0.82 ± 0.10 at 800 (n = 7), "
            "0.81 ± 0.09 at 1200 (n = 4). At 40 min: 1.64 ± 0.46, 2.23 ± 0.49, "
            "and 3.76 ± 0.66. Their folds versus rest: approximately 2-, 3-, "
            "and 5-fold at 400, 800, and 1200 kg-m/min (1 kg-m/min = 0.163 W)."
        ),
        marker_id="hepatic_output",
        blocked_on="m2.1",
        note=(
            "The number is filed. It still cannot train until the "
            "sympathoadrenal arm exists."
        ),
    ),
    QueueItem(
        id="fat-slows-emptying",
        citation="A carb-only load versus an equicaloric high-fat load",
        status=Status.BLOCKED,
        surface="cohort",
        claim="Fat in a meal slows glucose appearance relative to the same carbohydrate alone.",
        marker_id="glucose",
        blocked_on="m2.2",
        note="Needs a stomach. The citation is the finding; the primary paper is picked when the hub is built.",
    ),
    QueueItem(
        id="glp1-delays-emptying",
        citation="Nauck et al. — GLP-1 delays gastric emptying",
        status=Status.BLOCKED,
        surface="cohort",
        claim="Raised GLP-1 slows gastric emptying and flattens the glucose curve.",
        marker_id="glucose",
        blocked_on="m2.2",
    ),
    QueueItem(
        id="nuttall-protein-insulin",
        citation="Nuttall & Gannon — protein meals and insulin",
        status=Status.BLOCKED,
        surface="cohort",
        claim="A protein meal raises insulin with little rise in glucose.",
        marker_id="insulin",
        blocked_on="m2.5",
        note="Secretion is gated by relu(G − Gb), so amino acids have no direct path to insulin.",
    ),
    QueueItem(
        id="gip-mixed-meal",
        citation="Mixed-meal GIP peak",
        status=Status.BLOCKED,
        surface="cohort",
        claim="GIP rises after a mixed meal with a published peak and timing.",
        marker_id=None,
        blocked_on="m2.5",
        note="No GIP state. Pick the paper's peak when the state is added.",
    ),
    QueueItem(
        id="glp1-suppresses-glucagon",
        citation="GLP-1 suppresses glucagon during a meal",
        status=Status.BLOCKED,
        surface="cohort",
        claim="A GLP-1 elevation lowers meal-time glucagon beyond what glucose and insulin do.",
        marker_id="glucagon",
        blocked_on="m2.5",
    ),
    QueueItem(
        id="bile-modulates-fat-absorption",
        citation="Intestinal bile acids and fat absorption",
        status=Status.BLOCKED,
        surface="prior",
        claim="Intestinal bile changes how fast meal fat is absorbed.",
        marker_id="intestinal_bile",
        blocked_on="m2.6",
        note="Four hepatobiliary states are supervised and nothing reads them.",
    ),
    QueueItem(
        id="glycosuria-threshold",
        citation="Renal glucose threshold, about 180 mg/dL",
        status=Status.BLOCKED,
        surface="cohort",
        claim="Urinary glucose appears above the renal threshold and is absent below it.",
        marker_id="glucose",
        blocked_on="m2.7",
    ),
    QueueItem(
        id="training-block-sensitivity",
        citation="Weeks of training and insulin sensitivity",
        status=Status.BLOCKED,
        surface="cohort",
        claim="A training block raises insulin sensitivity and lowers resting heart rate.",
        marker_id="hr",
        blocked_on="m2.9",
        note="The current teacher moves mitochondrial capacity +0.8% over 8 weeks and resting HR not at all.",
    ),
    QueueItem(
        id="ogtt-insulin-120min",
        citation="PMID 3525269 — 75 g OGTT, 247 healthy subjects with normal glucose tolerance",
        status=Status.INTEGRATED,
        surface="cohort",
        claim=(
            "Two-hour insulin is 0.24 ± 0.01 nmol/L (SEM), 34.6 ± 1.44 µU/mL "
            "at 6.945 pmol/L per µU/mL. The window is 110–130 min after the load."
        ),
        marker_id="insulin",
        spec_name="ogtt_75g_insulin_120min",
        note=(
            "Sigma is the published SEM, n_arm=1. The three-hour mean can be "
            "met by a peak and a crash; this window is the tail, and the "
            "potentiator is what can hold insulin up while glucose falls."
        ),
    ),
    QueueItem(
        id="landau-gng-share",
        citation="Landau et al. (1996) — gluconeogenic contribution to glucose production",
        status=Status.BLOCKED,
        surface="cohort",
        claim=(
            "Post-absorptive gluconeogenesis is about half of endogenous "
            "glucose production. The prolonged-fast floor is gluconeogenesis alone."
        ),
        marker_id="hepatic_output",
        blocked_on="m1.5",
        note=(
            "f_gng is a population parameter under a Beta prior centred at one half. "
            "Landau's 47% / 67% / 93% at 14 h / 22 h / 42 h is a trajectory of the "
            "split, and hepatic_output is only the sum, so a spec still cannot see it."
        ),
    ),
    QueueItem(
        id="eaton-ffa-clearance",
        citation="Eaton et al. 1969 — plasma FFA half-time about 2–4 min",
        status=Status.INTEGRATED,
        surface="prior",
        claim=(
            "FFA clearance is a few minutes. The prior is log-normal, median "
            "the teacher's k_ffa = 0.20 (half-time 3.5 min), and the log-sigma "
            "is half of Eaton's factor-of-two between 2 and 4 min."
        ),
        marker_id="ffa",
        spec_name="k_ffa",
    ),
    QueueItem(
        id="cole-1999-hrr1",
        citation="Cole et al. (1999) N Engl J Med 341:1351 — PMID 10536127",
        status=Status.BACKLOG,
        surface="cohort",
        claim=(
            "Heart-rate recovery, peak to one minute later, in 2428 referred "
            "adults (mean age 57) during a cool-down: median 17 bpm, "
            "25th–75th percentile 12–23. A fall of 12 bpm or less was the "
            "mortality cutoff, not the healthy mean."
        ),
        marker_id="hr",
        note=(
            "Heart rate and activity exist, and the number is real. The "
            "protocol is symptom-limited peak plus a 1.5 mph cool-down in a "
            "referred cohort. The moderate-bout arm is not that protocol, "
            "and 17 bpm must not be aimed at the median person. Becomes ready "
            "when an arm is that protocol, or when a healthy-subject one-minute "
            "recovery paper is filed in its place. hr_recovery_post_exercise "
            "is a different claim: ≥20 bpm over 30 minutes."
        ),
    ),
)


def ready_items() -> tuple[QueueItem, ...]:
    return tuple(item for item in QUEUE if item.status is Status.READY)


def blocked_items() -> tuple[QueueItem, ...]:
    return tuple(item for item in QUEUE if item.status is Status.BLOCKED)
