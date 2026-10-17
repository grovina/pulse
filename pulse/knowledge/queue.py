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
    "m2.4",  # energy expenditure
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
            "Lipolysis is multiplied by epinephrine above basal, and that "
            "gain is a parameter. This rule is what moves it."
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
        note=(
            "Both models now have sns_tone and epinephrine. Epinephrine's "
            "drive starts at 69 mg/dL and cortisol's hypoglycaemic drive at "
            "58. The ranked hinge still cannot train: an arm has no insulin "
            "clamp, and a day without food stays near 83 mg/dL. Growth "
            "hormone has no state."
        ),
    ),
    QueueItem(
        id="donga-2010-sleep-clamp",
        citation="Donga et al. (2010) J Clin Endocrinol Metab 95:2963, Table 1",
        status=Status.INTEGRATED,
        surface="cohort",
        claim=(
            "One night, sleep allowed 01:00–05:00 versus 23:00–07:30, both "
            "centered at 03:00, n = 9. Basal glucose the next morning is "
            "4.8 ± 0.18 mmol/L on both occasions. At the clamp, glucose "
            "disposal is 32.5 ± 3.6 versus 40.7 ± 5.1 µmol/kg LBM/min, "
            "endogenous production 4.4 ± 0.3 versus 3.6 ± 0.2, and NEFA "
            "68 ± 5 versus 57 ± 4 µmol/L. Basal endogenous production is unchanged."
        ),
        marker_id="glucose",
        spec_name="donga_2010_basal_glucose",
        note=(
            "The cohort is the basal glucose row, which is the comparison a "
            "rollout can score. Disposal, clamp production and clamp NEFA are "
            "the debt factors in circadian.py, checked against the table at "
            "22/3.1 times basal insulin. There is no insulin clamp in a rollout."
        ),
    ),
    QueueItem(
        id="wahren-1971-exercise-hgo",
        citation="Wahren et al. (1971) J Clin Invest — splanchnic glucose output in exercise",
        status=Status.INTEGRATED,
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
        spec_name="wahren_800_splanchnic_glucose",
        note=(
            "The 800 kg-m/min load, on activity 0.7, as a rise in "
            "hepatic_output. The 400 and 1200 loads are not separate specs. "
            "Arterial glucose on the same bout is wahren_800_arterial_glucose."
        ),
    ),
    QueueItem(
        id="wahren-1971-exercise-glucose",
        citation="Wahren et al. (1971) J Clin Invest — arterial glucose in exercise",
        status=Status.INTEGRATED,
        surface="cohort",
        claim=(
            "Same men and the same 800 kg-m/min bout. Arterial glucose, "
            "mmol/L, mean ± SE, n = 9: rest 4.16 ± 0.12, 40 min 4.62 ± 0.33."
        ),
        marker_id="glucose",
        spec_name="wahren_800_arterial_glucose",
        note=(
            "The rise, 8.28 ± 6.32 mg/dL, on the splanchnic spec's arms. "
            "Together they are what move exercise uptake."
        ),
    ),
    QueueItem(
        id="fat-slows-emptying",
        citation="Cunningham & Read, Br J Nutr 1989;61:285 — fat in a meal and the glucose curve",
        status=Status.BLOCKED,
        surface="cohort",
        claim="Fat in a meal slows glucose appearance relative to the same carbohydrate alone.",
        marker_id="glucose",
        blocked_on="m2.2",
        note=(
            "The compartment is in (Brener 1983, 2.13 kcal/min): 50 g of "
            "carbohydrate appears at 4.11 mg/dL/min and with 30 g of fat at "
            "1.75. This item stays blocked until a published glucose or "
            "emptying number is the cohort target. The peaks quoted from "
            "Cunningham were not in the paper's table."
        ),
    ),
    QueueItem(
        id="glp1-delays-emptying",
        citation="Nauck et al. — GLP-1 delays gastric emptying",
        status=Status.BLOCKED,
        surface="cohort",
        claim="Raised GLP-1 slows gastric emptying and flattens the glucose curve.",
        marker_id="glucose",
        blocked_on="m2.2",
        note=(
            "The stomach is in, and its rate is fixed from the meal before "
            "the step. A GLP-1 brake has to read the hormone inside the "
            "minute, which this compartment does not."
        ),
    ),
    QueueItem(
        id="nuttall-protein-insulin",
        citation="Westphal, Gannon and Nuttall (1990) Am J Clin Nutr 52:267",
        status=Status.INTEGRATED,
        surface="cohort",
        claim=(
            "50 g of protein alone does not change serum glucose. Insulin "
            "and glucagon both rise."
        ),
        marker_id="glucose",
        spec_name="westphal_1990_protein_glucose",
        note=(
            "The cohort is the glucose row. Fasting glucose was 4.8 ± 0.1 "
            "mmol/L and the difference was not published, so sigma is that "
            "SEM carried as an unpaired SE. The insulin rise is the amino-acid "
            "term in incretin.py; the figure's µU/mL was not a table."
        ),
    ),
    QueueItem(
        id="gip-mixed-meal",
        citation="Vollmer et al. (2008) Diabetes 57:678",
        status=Status.BLOCKED,
        surface="cohort",
        claim=(
            "Integrated GIP after an 820 kcal breakfast is 186 ± 17% higher "
            "than after 75 g of glucose."
        ),
        marker_id="gip",
        blocked_on="m2.5",
        note=(
            "The gip state is in. The ratio is pinned in tests/test_incretin.py. "
            "A cohort needs a pmol/L peak, and the paper's concentrations are "
            "in the figures."
        ),
    ),
    QueueItem(
        id="glp1-suppresses-glucagon",
        citation="Salehi, Vahl and D'Alessio (2008) J Clin Endocrinol Metab 93:4909",
        status=Status.BLOCKED,
        surface="cohort",
        claim=(
            "With glucose clamped, exendin-(9-39) raises the incremental "
            "glucagon area after 75 g of oral glucose from 221 ± 353 to "
            "1657 ± 699 pg/mL·min."
        ),
        marker_id="glucagon",
        blocked_on="m2.5",
        note=(
            "The hold is in incretin.py: at the GLP-1 excess of a 75 g load "
            "while it is emptying, equilibrium glucagon is 8.2 pg/mL lower "
            "than with GLP-1 at basal, and the term is zero at basal. The "
            "protocol is a glucose clamp, which a rollout cannot run. "
            "ogtt_glucagon_suppression still scores the free load."
        ),
    ),
    QueueItem(
        id="bile-modulates-fat-absorption",
        citation="Intestinal bile acids and fat absorption",
        status=Status.BLOCKED,
        surface="prior",
        claim="Intestinal bile changes how fast meal fat is absorbed.",
        marker_id="intestinal_bile",
        blocked_on="m2.6",
        note=(
            "The floor is in bile.py: an empty intestine absorbs 65% of the "
            "fat that leaves the stomach, and the resting content absorbs "
            "all of it. A fistula arm is not a rollout. Wu 2013 shows "
            "jejunal taurocholate raises GLP-1 only while glucose is "
            "absorbed, and the abstract gives no concentration."
        ),
    ),
    QueueItem(
        id="glycosuria-threshold",
        citation="Renal glucose threshold, about 180 mg/dL",
        status=Status.BLOCKED,
        surface="cohort",
        claim="Urinary glucose appears above the renal threshold and is absent below it.",
        marker_id="glucose",
        blocked_on="m2.7",
        note=(
            "The threshold and the clearance are in renal.py: nothing "
            "leaves at or below 190 mg/dL, and ten above it leaves at "
            "13.9 mg/min. A rollout does not cross the threshold, and "
            "there is no clamp arm. Lactate returning as glucose is still "
            "unspecified: the share has no number yet."
        ),
    ),
    QueueItem(
        id="empagliflozin-24h-uge",
        citation="Diabetes 2017;66:1999–2006, day-13 urine, people without diabetes",
        status=Status.BLOCKED,
        surface="cohort",
        claim="Empagliflozin 25 mg/day raises a day's urinary glucose to 42 ± 5 g.",
        marker_id="glucose",
        blocked_on="m2.8",
        note=(
            "The day-14 regression intercept (34 mg/dL) and the day-13 "
            "clearance (126 mL/min) are the input in renal.py. A fed day "
            "of this model loses about 104 g, and the line is not retuned "
            "to the free-living collection. Urine is not a marker, so the "
            "collection has no cohort row."
        ),
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
