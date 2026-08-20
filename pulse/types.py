from dataclasses import dataclass
from enum import Enum


class System(str, Enum):
    METABOLIC = "metabolic"
    APPETITE = "appetite"
    STRESS = "stress"
    CARDIOVASCULAR = "cardiovascular"
    THERMOREG = "thermoreg"
    RESPIRATORY = "respiratory"
    HEPATOBILIARY = "hepatobiliary"


@dataclass(frozen=True)
class MarkerDef:
    id: str
    name: str
    unit: str
    system: System
    typical: float
    # API / frontend (generated into TS — keep in sync via scripts/generate_markers.py)
    ui_system: str
    min: float
    max: float
    color: str
    icon: str


@dataclass(frozen=True)
class SubjectiveSignalDef:
    id: str
    name: str


@dataclass(frozen=True)
class CouplingEdge:
    """A directed connection between modules in the coupling graph.

    sign_prior encodes medical knowledge about the direction of the effect:
    +1 means the source marker promotes/increases the target's activity,
    -1 means it suppresses/decreases it. This is used for initialization
    and regularization, not as a hard constraint.
    """
    source_module: str
    source_marker: str
    target_module: str
    sign_prior: int


MARKERS = [
    # Metabolic (indices 0-6): blood chemistry + hepatic glucose output (latent flux)
    MarkerDef(
        "glucose", "Blood Glucose", "mg/dL", System.METABOLIC, 95,
        "metabolic", 40, 400, "#f59e0b", "Droplets",
    ),
    MarkerDef(
        "insulin", "Insulin", "μU/mL", System.METABOLIC, 10,
        "metabolic", 0, 200, "#f97316", "Syringe",
    ),
    MarkerDef(
        "glucagon", "Glucagon", "pg/mL", System.METABOLIC, 70,
        "metabolic", 0, 500, "#eab308", "FlaskConical",
    ),
    MarkerDef(
        "ffa", "Free Fatty Acids", "mmol/L", System.METABOLIC, 0.5,
        "metabolic", 0, 3, "#d97706", "Flame",
    ),
    MarkerDef(
        "bhb", "β-Hydroxybutyrate", "mmol/L", System.METABOLIC, 0.1,
        "metabolic", 0, 10, "#ca8a04", "Zap",
    ),
    MarkerDef(
        "lactate", "Lactate", "mmol/L", System.METABOLIC, 1.0,
        "metabolic", 0, 20, "#a16207", "Activity",
    ),
    MarkerDef(
        "hepatic_output",
        "Hepatic glucose output",
        "mg/min",
        System.METABOLIC,
        2.0,
        "metabolic",
        0,
        12,
        "#b45309",
        "Gauge",
    ),
    # Appetite (indices 7-9): hunger/satiety hormones with mass-action kinetics
    MarkerDef(
        "ghrelin", "Ghrelin", "pg/mL", System.APPETITE, 100,
        "hormonal", 20, 300, "#a78bfa", "Utensils",
    ),
    MarkerDef(
        "leptin", "Leptin", "ng/mL", System.APPETITE, 10,
        "hormonal", 1, 50, "#8b5cf6", "Scale",
    ),
    MarkerDef(
        "glp1", "GLP-1", "pmol/L", System.APPETITE, 10,
        "hormonal", 2, 100, "#7c3aed", "Pill",
    ),
    # Stress (indices 10-11): HPA axis — ACTH drives cortisol; cortisol feeds back
    MarkerDef(
        "cortisol", "Cortisol", "μg/dL", System.STRESS, 12,
        "hormonal", 1, 40, "#6d28d9", "Brain",
    ),
    MarkerDef(
        "acth", "ACTH", "pg/mL", System.STRESS, 30,
        "hormonal", 5, 120, "#5b21b6", "Zap",
    ),
    # Cardiovascular (indices 12-15): vitals with learned dynamics
    MarkerDef(
        "hr", "Heart Rate", "bpm", System.CARDIOVASCULAR, 70,
        "vital", 30, 220, "#f43f5e", "Heart",
    ),
    MarkerDef(
        "hrv", "HRV (RMSSD)", "ms", System.CARDIOVASCULAR, 40,
        "vital", 0, 300, "#e11d48", "HeartPulse",
    ),
    MarkerDef(
        "sbp", "Systolic BP", "mmHg", System.CARDIOVASCULAR, 120,
        "vital", 60, 250, "#be123c", "ArrowUp",
    ),
    MarkerDef(
        "dbp", "Diastolic BP", "mmHg", System.CARDIOVASCULAR, 80,
        "vital", 30, 150, "#9f1239", "ArrowDown",
    ),
    # Thermoregulation (index 16): learned dynamics
    MarkerDef(
        "temp", "Core Temperature", "°C", System.THERMOREG, 37.0,
        "vital", 35, 42, "#3b82f6", "Thermometer",
    ),
    # Respiratory (indices 17-18): learned dynamics
    MarkerDef(
        "rr", "Respiratory Rate", "/min", System.RESPIRATORY, 15,
        "vital", 4, 40, "#6366f1", "Wind",
    ),
    MarkerDef(
        "spo2", "SpO₂", "%", System.RESPIRATORY, 98,
        "vital", 70, 100, "#8b5cf6", "Percent",
    ),
    # Slow-timescale internal state in the metabolic module (iter 55+).
    # Unobserved (no direct measurement); supervised via cohort specs +
    # physiology rules from training-physiology literature. Indices 19+
    # placed at end of MARKERS so other markers' indices are preserved.
    # marker_type="internal" signals to UI / scoring that these have no
    # ground truth and should be skipped in observation-driven losses.
    # See docs/multi-timescale-plan.md.
    #
    # Iter 56: the iter-55 lumped `glycogen_pool` (500 g, τ ≈ 10 d) could
    # not express a −60 g / 1-day fast delta — one cons_scale cannot
    # serve both the fast-depleting liver pool and the slow muscle pool.
    # Split by tissue (anatomy as structure): liver_glycogen is the
    # overnight-depleting fast pool (τ ≈ 1 d); muscle_glycogen is the
    # slow exercise-coupled reservoir (τ ≈ weeks; chronic-exercise
    # north star).
    MarkerDef(
        "liver_glycogen",
        "Liver glycogen",
        "g",
        System.METABOLIC,
        100.0,
        "internal",
        0,
        150,
        "#92400e",
        "Battery",
    ),
    MarkerDef(
        "muscle_glycogen",
        "Muscle glycogen",
        "g",
        System.METABOLIC,
        400.0,
        "internal",
        0,
        600,
        "#b45309",
        "Battery",
    ),
    MarkerDef(
        "mitochondrial_capacity",
        "Mitochondrial capacity",
        "× pop. mean",
        System.METABOLIC,
        1.0,
        "internal",
        0,
        3,
        "#78350f",
        "Atom",
    ),
    # Iter 69 "Move B FULL" — CRH as a latent first stage of the HPA
    # cascade (CRH → ACTH → cortisol). The iter 64 prior collapsed the
    # cascade into ACTH(diurnal) → cortisol; iter 66 added the cortisol
    # cosinor. Both still place the entire delay structure in a single
    # 1-stage rate equation, so the model has no way to express the
    # Gamma-shape transduction kernel that the literature actually
    # characterises (CRH peaks lead ACTH by ~10-20 min, ACTH peaks lead
    # cortisol by ~10-15 min — two sequential first-order stages =
    # Gamma shape 2). Adding CRH as a state variable lets the cascade
    # express that delay structurally; cortisol's negative feedback now
    # hits CRH (anatomically correct — the dominant negative feedback
    # target is the hypothalamic PVN, not the corticotrophs), and the
    # diurnal drive enters CRH (still anatomically correct — SCN→PVN
    # is the canonical pathway). Internal/unobserved: the cold model
    # pads typical (no CRH simulation), the learned model discovers
    # its shape from the cascade-derived ACTH/cortisol matches.
    MarkerDef(
        "crh",
        "CRH",
        "pg/mL",
        System.STRESS,
        100.0,
        "internal",
        0,
        500,
        "#9f1239",
        "Wave",
    ),
    # Iter 89 — dynamic insulin action (remote insulin) as a latent metabolic
    # state. The teacher (full_body.py) drives glucose clearance with a LAGGED
    # remote-insulin state X (dX = -p2·X + p3·max(I-Ib,0), τ ≈ 33 min, Bergman
    # minimal model), but through iter 88 the student's clearance used an
    # INSTANTANEOUS x_ins = Si·relu(insulin) — no delay, so meal glucose fell
    # too fast relative to the teacher. This state gives the student the same
    # first-order lag: insulin_action low-passes relu(insulin_above_baseline),
    # and glucose clearance reads the lagged state instead of instantaneous
    # insulin. Internal/unobserved (no ground truth): the cold model pads it at
    # typical=0 (fasting equilibrium, where relu(insulin-baseline)=0, so glucose
    # dynamics are byte-identical to iter 88 at rest); the lag manifests only
    # during meals. Appended at the tail so all prior marker indices are
    # preserved (mirrors crh iter-69). NORM_SCALE 1.0 so raw == normalized.
    MarkerDef(
        "insulin_action",
        "Insulin action (remote)",
        "a.u.",
        System.METABOLIC,
        0.0,
        "internal",
        0,
        5,
        "#c2410c",
        "Timer",
    ),
    # --- Hepatobiliary (iter 95) -------------------------------------------
    # The enterohepatic circulation, entered at the MEDIATOR rather than at the
    # damage readouts. Full design + sourced anchors: docs/iter95-biliary-anchors.md.
    #
    #   meal lipid/protein -> [cck] -gate-> [gallbladder_bile] -> [intestinal_bile]
    #                              ^                                     |
    #                     canalicular export                    ileal reabsorption ~95%
    #                              |                                     v
    #                        hepatocyte <----- portal return ------------+
    #                              |
    #                     first-pass extraction -> spillover -> [bile_acids]
    #
    # Why here and not at ALP/GGT/ALT/bilirubin: those are damage and obstruction
    # readouts, and with no injury or obstruction in the state vector they would ship
    # as constants — and the iter-94 ruler finding is precisely that the gate cannot
    # tell a constant from a simulator. Cholestasis IS failure of canalicular export,
    # so modelling that step explicitly means the enzymes later hang off impairment of
    # a step that already exists instead of needing a driver bolted on.
    #
    # Appended at the tail (mirrors crh iter-69 / insulin_action iter-89) so all prior
    # marker indices are preserved.
    MarkerDef(
        # Duodenal I-cell secretion in response to intraluminal fat and protein — which
        # is why the gut module's EXISTING lipid and amino appearance channels are the
        # stimulus and no new model input is needed. Fasting 0.8-1.2 pmol/L, peak
        # 6.5-7.1 within ~10 min of a mixed meal, ~3.5 at 30 min, elevated 3-5 h.
        "cck", "Cholecystokinin", "pmol/L", System.HEPATOBILIARY, 1.0,
        "internal", 0, 30, "#65a30d", "Zap",
    ),
    MarkerDef(
        # The axis's one genuine POOL. Content in mmol rather than volume in mL: the
        # gallbladder concentrates bile ~10x, so content is what conserves around the
        # loop, and the two are interchangeable up to a concentration constant.
        # Contraction is a GATE computed from cck, not a state — see
        # docs/iter95-proposal.md 3.2.1 for why the reverse parameterisation is wrong
        # (you cannot empty a gallbladder twice). Ejection fraction is DERIVED and
        # scored against the >=35-38% HIDA threshold.
        "gallbladder_bile", "Gallbladder bile acids", "mmol",
        System.HEPATOBILIARY, 4.0,
        "internal", 0, 12, "#4d7c0f", "Droplet",
    ),
    MarkerDef(
        # Carries the transit delay AND the 95%/5% reabsorption split. This state is
        # what makes the CCK-peak-at-10-min vs serum-peak-at-75-120-min gap a
        # CONSEQUENCE of transport in series rather than a fitted lag constant, and it
        # turns reabsorption efficiency into a mass balance rather than a number.
        "intestinal_bile", "Intestinal bile acids", "mmol",
        System.HEPATOBILIARY, 1.0,
        "internal", 0, 10, "#3f6212", "ArrowDownUp",
    ),
    MarkerDef(
        # The observable. Fasting reference 4.4-14.1 umol/L, postprandial 4.7-20.2,
        # rise beginning within 30 min and peaking 75-120 min. Only a small spillover
        # of the portal return escapes hepatic first-pass extraction — which is exactly
        # why impaired canalicular export raises it.
        "bile_acids", "Serum total bile acids", "µmol/L",
        System.HEPATOBILIARY, 6.0,
        "internal", 0, 100, "#a3e635", "Waves",
    ),
]

STATE_DIM = len(MARKERS)
# iter 60: widened 32->64 after glp1/ghrelin collapsed to the population mean. But the REAL
# fix in that iter was widening the AppetiteModule's PROJECTION (6 -> 16 dims); the shared
# embedding was widened alongside it, and that half was never needed.
#
# Iter 91: back to 32, because the shared embedding is the SIZE OF THE INVERSE PROBLEM.
# Calibration finds a new person by solving for this vector from their observations. A
# calibration window supplies ~5 markers x ~10 check-ins ~= 50 numbers. With 64 unknowns the
# system is UNDERDETERMINED -- infinitely many embeddings explain the same data -- and that is
# exactly what we measure: the recovered embedding is orthogonal to the truth (cos ~= 0) while
# fitting the observations, and Gb recovery is worse than predicting the population mean.
#
# The extra dimensions buy nothing. Measured (SetpointSupervisionSignal fitting 10 patients'
# Gb/HR0/SBP0/DBP0 to convergence, module projections unchanged):
#     EMB_DIM   glucose_mae   hr_mae   sbp_mae   dbp_mae
#        64        0.006       0.020     0.908     0.274
#        32        0.013       0.027     0.870     0.285   <-- same fit, half the unknowns
#        24        0.008       0.029     1.071     0.324
# So dims 33-64 are free parameters the observations cannot pin down: pure underdetermination
# with no capacity benefit. This is also what the teacher says -- it generates each patient from
# TWO latent axes (insulin-resistance, fitness) plus per-parameter noise, so the true patient
# manifold is low-dimensional.
#
# Module capacity is UNCHANGED: the per-module projections (EMBEDDING_DIM -> {gut 16,
# metabolic 20, appetite 16, stress 12, cardiovascular 16, thermoreg 8, respiratory 8}) keep
# their widths, so iter-60's actual fix survives intact.
EMBEDDING_DIM = 32
TIME_FEATURES_DIM = 3
EXTERNAL_INPUT_DIM = 2  # sleep_wake, activity

# Gut module outputs (not part of the ODE state)
GUT_OUTPUT_DIM = 4  # glucose_appearance, lipid_appearance, amino_appearance, nutrient_flag

MARKER_IDS = [m.id for m in MARKERS]
MARKER_INDEX = {m.id: i for i, m in enumerate(MARKERS)}

NORM_CENTER = [m.typical for m in MARKERS]

NORM_SCALE = [
    30.0,   # glucose: ±30 mg/dL during meals
    10.0,   # insulin: ±10 μU/mL during meals
    20.0,   # glucagon: ±20 pg/mL
    0.2,    # ffa: ±0.2 mmol/L
    # bhb: ±0.5 mmol/L (iter 95; was ±0.05). BHB spans 0.1 mmol/L fed to 1-2 by a 24 h
    # fast and 3-5 in sustained ketosis (Cahill 2006), so ±0.05 was not a typical
    # excursion — it was ~1/20th of one, and it broke two things at once.
    #   1. PHYSIOLOGICAL_MAX = center + 20·NORM_SCALE landed at 1.10 mmol/L — BELOW the
    #      teacher's own 1.315 at a 24 h fast, and far below its 3.505 at 48 h. The
    #      catastrophe clamp sat inside the physiological range, so even a perfect
    #      student would have been clipped. It is now 10.1.
    #   2. Every metabolic head reads the whole module state, normalized. At ±0.05 a
    #      ketotic BHB is z ≈ +24 while every other species lives in z ∈ [-2, +2], so a
    #      WORKING bhb would have driven every head's tanh layers into saturation. This
    #      was invisible only because bhb was frozen at typical (the absorbing floor —
    #      see modules/base.py:MassActionModule); fixing that floor without fixing this
    #      scale would have traded one failure for another.
    # Note this also reduces bhb's weight in every NORM_SCALE-normalized loss by 10x.
    # That is the point: at ±0.05 a 1.2 mmol/L residual counted as 24 z-units, dominating
    # the distillation objective for a marker that could not move at all.
    0.5,    # bhb: ±0.5 mmol/L
    0.3,    # lactate: ±0.3 mmol/L
    1.2,    # hepatic_output: endogenous appearance flux (not directly measured)
    40.0,   # ghrelin: ±40 pg/mL (preprandial rise/postprandial drop)
    3.0,    # leptin: ±3 ng/mL (slow circadian variation)
    15.0,   # glp1: ±15 pmol/L (postprandial spikes)
    8.0,    # cortisol: ±8 μg/dL (circadian range)
    18.0,   # acth: ±18 pg/mL (circadian + stress)
    10.0,   # hr: ±10 bpm
    15.0,   # hrv: ±15 ms
    10.0,   # sbp: ±10 mmHg
    8.0,    # dbp: ±8 mmHg
    0.3,    # temp: ±0.3 °C
    3.0,    # rr: ±3 /min
    1.5,    # spo2: ±1.5%
    60.0,   # liver_glycogen: ±60 g (typical ~100 g; ~16h fast depletes ~60 g of the liver pool)
    100.0,  # muscle_glycogen: ±100 g (typical ~400 g; rest-preserved over 1-day fast, depletes with exercise)
    0.3,    # mitochondrial_capacity: ±0.3 (typical=1.0; 8w training adds ~30%, so ±0.3 covers training-induced range)
    40.0,   # crh: ±40 pg/mL (typical=100; latent first stage of HPA cascade, circadian + stress amplitude)
    1.0,    # insulin_action: dimensionless latent (raw==normalized); low-passes relu(insulin_norm) ∈ ~[0,5]
    # --- Hepatobiliary (iter 95). Each is a TYPICAL EXCURSION, not a reference range:
    # the iter-95 bhb lesson is that a NORM_SCALE far below the marker's real swing puts
    # the catastrophe clamp inside physiology and drives every head's tanh into
    # saturation once the marker actually moves.
    2.0,    # cck: ±2 pmol/L (typical 1.0; postprandial peak 6.5-7.1, so a meal is ~+3σ)
    1.5,    # gallbladder_bile: ±1.5 mmol (typical 4.0; a >=35% ejection is ~-1σ)
    1.0,    # intestinal_bile: ±1.0 mmol (typical 1.0; fills to several mmol after emptying)
    4.0,    # bile_acids: ±4 µmol/L (typical 6.0; fasting interval 4.4-14.1, postprandial to 20.2)
]

# Hard physiological bounds for the integrator's state clamp (iter 81).
#
# These are NOT the clinical-normal ranges — they are the "incompatible with
# continued physiology / measurement saturation" extremes, used purely as a
# catastrophe safety on the forward integration. The model integrates raw state
# with plain forward Euler and no clamp (model.py), and every module input
# includes the raw per-patient embedding. A calibrated embedding that wanders
# off the trained manifold (benchmark calibration is only weakly leashed) can
# drive an unbounded softplus production term or the gut appearance kernel to
# nonphysical magnitudes — observed: gut glucose appearance ~500x typical →
# glucose integrating to ~18,000 mg/dL. Clamping the integrated state to these
# extremes makes such a trajectory bounded (and the prediction merely wrong, not
# catastrophic) for every marker regardless of which internal term diverged.
#
# Anchored to the model's OWN scale (NORM_CENTER ± K·NORM_SCALE), floored at 0
# (no marker is physically negative). K is deliberately large so the clamp is
# INACTIVE in-distribution and only catches divergence. NOT anchored to the
# clinical [min, max]: the internal/unobserved markers (liver_glycogen,
# muscle_glycogen, …) carry no ground truth, so the model learned its own scale
# for them that need not match the display range — measured, in-distribution
# liver_glycogen reaches ~12 SD (NORM_SCALE underestimates its true swing),
# which a clinical-range bound would wrongly clip. The model's z-score envelope
# over the calibration-reachable region (||emb||≤3, fed + 24h-fast protocols)
# peaks at ~12.5 SD; K=20 clears that with margin so nothing in-distribution is
# touched, while still bounding a runaway (e.g. glucose to ≤95+20·30=695 mg/dL,
# vs the ~18,000 observed unclamped). The real fix for off-manifold calibration
# is the embedding leash in benchmark.py — this is defense-in-depth that also
# keeps training rollouts from exploding.
PHYSIOLOGICAL_LIMIT_K = 20.0

PHYSIOLOGICAL_MIN = [
    max(0.0, c - PHYSIOLOGICAL_LIMIT_K * s)
    for c, s in zip(NORM_CENTER, NORM_SCALE)
]
PHYSIOLOGICAL_MAX = [
    c + PHYSIOLOGICAL_LIMIT_K * s
    for c, s in zip(NORM_CENTER, NORM_SCALE)
]

# Which markers belong to each module (as indices into the state vector)
MODULE_MARKER_INDICES: dict[str, list[int]] = {}
for system in System:
    MODULE_MARKER_INDICES[system.value] = [
        i for i, m in enumerate(MARKERS) if m.system == system
    ]

MODULE_MARKER_IDS: dict[str, list[str]] = {
    system: [MARKERS[i].id for i in indices]
    for system, indices in MODULE_MARKER_INDICES.items()
}

# The coupling graph — anatomical connections between modules.
# Existence is structural (architecture). Sign is a medical prior (regularized).
# Strength is learned from data.
COUPLING_GRAPH = [
    # Cortisol drives hepatic glucose output
    CouplingEdge("stress", "cortisol", "metabolic", sign_prior=+1),
    # Insulin suppresses ghrelin; nutrient sensing triggers GLP-1
    CouplingEdge("metabolic", "insulin", "appetite", sign_prior=-1),
    # Hypoglycemia triggers cortisol release
    CouplingEdge("metabolic", "glucose", "stress", sign_prior=-1),
    # Cortisol negative feedback on pituitary ACTH
    CouplingEdge("stress", "cortisol", "stress", sign_prior=-1),
    # Cortisol raises HR/BP via sympathetic activation
    CouplingEdge("stress", "cortisol", "cardiovascular", sign_prior=+1),
    # Temperature affects cardiovascular dynamics
    CouplingEdge("thermoreg", "temp", "cardiovascular", sign_prior=+1),
    # Cortisol and metabolic rate affect temperature
    CouplingEdge("stress", "cortisol", "thermoreg", sign_prior=+1),
    CouplingEdge("metabolic", "glucose", "thermoreg", sign_prior=+1),
    # Lactate (metabolic demand proxy) drives respiratory rate
    CouplingEdge("metabolic", "lactate", "respiratory", sign_prior=+1),
]

# Training observation intervals (minutes between samples)
OBS_INTERVALS = {
    "glucose": 5,
    "insulin": 120,
    "glucagon": 120,
    "ffa": 120,
    "bhb": 60,
    "lactate": 60,
    "hepatic_output": 120,
    "ghrelin": 60,
    "leptin": 360,
    "glp1": 60,
    "cortisol": 60,
    "acth": 60,
    "hr": 1,
    "hrv": 1,
    "sbp": 30,
    "dbp": 30,
    "temp": 15,
    "rr": 5,
    "spo2": 5,
}

SUBJECTIVE_SIGNALS = [
    SubjectiveSignalDef("hungry", "Feeling Hungry"),
    SubjectiveSignalDef("full", "Feeling Full"),
    SubjectiveSignalDef("stressed", "Feeling Stressed"),
    SubjectiveSignalDef("tired", "Feeling Tired"),
    SubjectiveSignalDef("shaky", "Feeling Shaky"),
]
