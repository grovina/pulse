from dataclasses import dataclass
from enum import Enum


class System(str, Enum):
    METABOLIC = "metabolic"
    APPETITE = "appetite"
    STRESS = "stress"
    CARDIOVASCULAR = "cardiovascular"
    THERMOREG = "thermoreg"
    RESPIRATORY = "respiratory"


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
]

STATE_DIM = len(MARKERS)
# iter 60: widened 32->64. The 32-dim patient code was the structural
# ceiling (spec R1 / architecture-roadmap.md #8): glp1/ghrelin (the two
# >100%-MAPE markers) collapsed to the population mean because the
# AppetiteModule's 6-dim projection could not encode two independent
# patient-specific spike manifolds. A 12x glp1 supervision bump (iter 59)
# moved it +0.0353 — the capacity-not-weight signature.
EMBEDDING_DIM = 64
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
    0.05,   # bhb: ±0.05 mmol/L
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
