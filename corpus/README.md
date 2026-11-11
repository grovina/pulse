# The corpus: one paper, one file

This directory is the model's reading list, kept as files. **Pulse** is a
differentiable model of human physiology (glucose, insulin, heart rate, blood
pressure, hormones, energy expenditure, and the organs that move them). It is
trained to agree with the medical literature. Each paper that reaches it
becomes one file here, and `python -m pulse.admit` decides whether the file may
enter.

You are handed one paper. You do not need to know how the model works inside.
You need to know three things: what the model can be told, how to write it
down, and when to say "it cannot be told this".

```
corpus/queue/<paper>.json      a paper waiting to be read: where its numbers are, no numbers
corpus/findings/<paper>.json   a paper the model can express: its numbers, as findings
corpus/refusals/<paper>.json   a paper it cannot yet express, and the one piece it is missing
```

**A paper is in exactly one of the three piles.** You are handed a paper from the
queue. When you file its finding or its refusal, **delete its `queue/` file in the same
commit**; the gate refuses a finding or a refusal whose paper is still queued, and a
test refuses a paper in two piles. `python -m pulse.admit --piles` counts them.

### The queue

One file per paper, so many readers can each take one without touching the same file.
It holds what a reader needs to start and nothing the reader must derive: no numbers,
and no `sealed` field (that follows from the slug and would go stale).

```json
{"paper": "horie-2018", "citation": "Horie I, et al. ... 2018.",
 "doi": "10.1111/jdi.12829", "pmid": "29489067", "pmcid": "PMC6215950",
 "licence": "CC BY-NC 4.0", "licence_url": "http://creativecommons.org/licenses/by-nc/4.0/",
 "protocol": "75 g OGTT after overnight fast; samples 0, 30, 60, 90, 120 min",
 "protocol_kinds": ["OGTT"], "population": "n=63 ... healthy Japanese adults",
 "population_class": "healthy",
 "reports": [{"marker": "glucose", "where": "Table 1 (0-120 min, by sex)", "form": "table"}],
 "uncertainty": "sd", "figure_only": false, "note": "free text"}
```

`doi`, `pmid`, `pmcid` and `licence_url` are always present, `null` when unknown, with at
least one identifier. `protocol_kinds` come from `OGTT, mixed meal, clamp, 24 h profile,
IVGTT, exercise, prolonged fast, other`; `population_class` is `healthy`, `nonhealthy` or
`mixed`; `reports[].form` is `table`, `text` or `figure`; `uncertainty` is `sd`, `sem`,
`ci`, `range` or `unlabelled`. `reports[].marker` is free text (a queued paper may report
what the model has no series for). The same DOI, PMID or PMCID in two files, or a slug
already in a Python registry, is refused. Full texts are not committed: the PMCID fetches
them again, and the licences differ.

`<paper>` is a lowercase slug, `firstauthor-year`: `schulla-2003`, `defronzo-1979`.
Where there is no author (a guideline, a reference range) use what it is:
`who-1999`. It is the file's name and the `paper` field, and the registry uses it
to decide, by a hash, whether this paper is **held out** of training and used
only to test the model. Nobody chooses that, and you cannot influence it except
by naming the paper truthfully. A slug ending in four digits must end in the
paper's year.

## What a finding is

A finding is one reported number, in a protocol, with a place in the paper and a
stated uncertainty. The model is a simulator: you describe what the subjects
did (the **protocol**: when they ate, slept, were infused), and what was
measured (the **marker**, over a **window** of minutes, summarised by a
**statistic**). The model rolls virtual people through the protocol and is
pulled toward the paper's number, as hard as the paper's uncertainty justifies.

Write only what the paper prints. Never fill a number from memory, never take
a "typical" value, never write a range as if it were a measurement. If the paper
gives a number only in a figure, write it only if you can read it off with a
stated uncertainty; otherwise leave it out. A number with no place in a paper is
not a finding.

### A complete example

This file is parsed by the test suite, so it cannot go stale. It is made up.

```json
{
  "paper": "example-2099",
  "citation": {
    "authors": "Example A, Example B",
    "title": "A made-up study used as the worked example",
    "journal": "Journal of Made-Up Examples",
    "year": 2099,
    "doi": "10.0000/example.2099"
  },
  "population": {
    "n": 12,
    "sex": "male",
    "age_years": {
      "mean": 25.0,
      "sd": 3.0,
      "min": 20.0,
      "max": 31.0
    },
    "bmi": null,
    "status": "healthy",
    "description": "Twelve made-up healthy young men."
  },
  "protocols": [
    {
      "name": "ogtt",
      "description": "75 g oral glucose after an overnight fast, two hours observed",
      "duration_min": 120,
      "start_hour": 8.0,
      "meals": [
        {
          "t_min": 0,
          "carbs_g": 75.0,
          "fat_g": 0.0,
          "protein_g": 0.0
        }
      ],
      "assumed": ["start_hour"]
    },
    {
      "name": "fasting",
      "description": "the same morning with water only",
      "duration_min": 120,
      "start_hour": 8.0,
      "meals": [],
      "assumed": ["start_hour"]
    }
  ],
  "findings": [
    {
      "id": "ogtt_glucose_peak",
      "description": "Peak plasma glucose in the first two hours of the OGTT",
      "marker": "glucose",
      "statistic": "peak_value",
      "protocols": [
        "ogtt"
      ],
      "window": {
        "start_min": 0,
        "end_min": 120
      },
      "value": 8.0,
      "unit": "mmol/L",
      "uncertainty": {
        "kind": "sd",
        "value": 1.2,
        "n": 12
      },
      "location": {
        "where": "Table 2, row 'peak'",
        "quote": "Peak glucose was 8.0 (SD 1.2) mmol/L (n = 12)."
      }
    },
    {
      "id": "ogtt_insulin_curve",
      "description": "Plasma insulin through the OGTT, mean and standard error",
      "marker": "insulin",
      "statistic": "curve",
      "protocols": [
        "ogtt"
      ],
      "unit": "pmol/L",
      "points": [
        {
          "t_min": 0,
          "value": 60.0,
          "uncertainty": {
            "kind": "sem",
            "value": 6.0,
            "n": 12
          }
        },
        {
          "t_min": 30,
          "value": 420.0,
          "uncertainty": {
            "kind": "sem",
            "value": 60.0,
            "n": 12
          }
        },
        {
          "t_min": 60,
          "value": 480.0,
          "uncertainty": {
            "kind": "sem",
            "value": 72.0,
            "n": 12
          }
        },
        {
          "t_min": 90,
          "value": 300.0,
          "uncertainty": {
            "kind": "sem",
            "value": 48.0,
            "n": 12
          }
        }
      ],
      "location": {
        "where": "Figure 1B",
        "quote": "Insulin (pmol/L) at 0, 30, 60 and 90 min: 60, 420, 480, 300 (mean +/- SEM)."
      }
    },
    {
      "id": "ogtt_minus_fasting_glucose",
      "description": "Mean glucose over the second hour, OGTT minus the water-only morning",
      "marker": "glucose",
      "statistic": "delta_means",
      "protocols": [
        "fasting",
        "ogtt"
      ],
      "window": {
        "start_min": 60,
        "end_min": 120
      },
      "value": 1.1,
      "unit": "mmol/L",
      "uncertainty": {
        "kind": "ci",
        "low": 0.5,
        "high": 1.7,
        "level": 95,
        "n": 12
      },
      "location": {
        "where": "Results, paragraph 3",
        "quote": "The second-hour difference was 1.1 mmol/L (95% CI 0.5 to 1.7)."
      }
    }
  ],
  "left_out": [
    {
      "what": "C-peptide at each sampling time (Table 2)",
      "missing": {"kind": "marker", "name": "c_peptide"}
    },
    {
      "what": "glucagon time course (Figure 3 only)",
      "missing": {"kind": "source", "name": "figure_only"}
    }
  ]
}
```

### The top level

| key | what it is |
|---|---|
| `paper` | the slug; also the file name |
| `citation` | `authors`, `title`, `journal`, `year`, and **at least one** of `doi`, `pmid`, `pmcid` (DOI as `10.xxxx/...`, PMID digits, PMCID `PMC123`). The same identifier under two slugs is refused: it is one paper. |
| `population` | who was measured, below |
| `protocols` | what they did, below |
| `findings` | the numbers, below |
| `left_out` | what the paper reports that you could not enter, and the piece that was missing; `[]` if nothing. Below. |

Every object is strict. An unknown key, a missing key, a wrong type, an unknown
marker or an unlisted unit is an error, and the gate lists **all** of a file's
errors together. Do not add keys; do not state `sigma`, `n_arm`, `weight`,
`shape` or `sealed`. They are computed or decided elsewhere.

### `population`

```json
{"n": 12, "sex": "male", "age_years": {"mean": 25.0, "sd": 3.0, "min": 20.0, "max": 31.0},
 "bmi": null, "status": "healthy", "description": "..."}
```

* `n` is the number of subjects the paper studied.
* `sex` is `female`, `male` or `mixed`.
* `age_years` and `bmi`: `null` if the paper does not report it. Otherwise all
  four of `mean`, `sd`, `min`, `max`, each a number or `null`, not all `null`.
  The virtual people are drawn from `[min, max]` if both are given, otherwise from
  `mean +/- 2 sd`.
  * `min` and `max` are the youngest and oldest (leanest and heaviest) subject
    **observed**. Enrolment criteria ("aged 18 to 40") are not that: put them in
    `description` and leave `min` and `max` `null`, unless the paper gives nothing
    else about age or BMI.
  * A median with an interquartile range is not a mean and an SD: `null`.
  * If the paper gives mean and standard error, `sd` is `SE x sqrt(n)`; say in
    `description` that you computed it. (This is the one place you convert. It
    only shapes who the virtual people are; no finding is scored against it.)
  * If the paper gives age or BMI for a larger or pooled group and not for the
    group the findings are about, use it and say so in `description`.
* `status` is `healthy` or `unselected` (no screening was reported). **Nothing
  else.** Diabetes, hypertension, pregnancy, an illness, a drug the model has no
  input for: the model has no way to say it. That is a refusal (kind `population`),
  not a finding with the wrong status. Obesity is not a condition here; it is a
  `bmi` range with `unselected`.
* A protocol can override `sex`, `age_years`, `bmi` or `status` for one arm, when
  the paper compares groups (lean against obese, young against old). `age_years`
  and `bmi` are written as at the top level, with that group's printed numbers
  (`{"mean": 57.0, "sd": 7.0, "min": null, "max": null}`), and must give a
  range: `min` and `max`, or `mean` and `sd`.
* Sex, age and BMI are all the model can be told about a group. Groups the paper
  separates by anything else (insulin-sensitive against insulin-resistant, high
  against low responders, trained against untrained) cannot be entered as two
  arms: the model would be asked for two different numbers from the same
  people. Enter the pooled group if the paper reports it, or the one group that
  is plainly the healthy reference, and leave the other out.
* A group chosen by a measurement cannot give a finding about that measurement.
  The two-hour glucose of people enrolled for a two-hour glucose below
  140 mg/dL says who was let in, not what the body does: leave it out
  (`population`, `group_split_by_measurement`). The ordinary screening of a
  healthy cohort (no diabetes, no medication) does not count.

### `protocols`

One entry per distinct arm of the paper's design that a finding uses. Every
protocol must be used by at least one finding.

| key | |
|---|---|
| `name` | lowercase letters, digits, underscore |
| `description` | one line |
| `duration_min` | minutes simulated: the arm has a state at minutes 0 to `duration_min - 1`. A test sampled at 0 and 120 minutes needs 121. |
| `start_hour` | clock hour at minute 0 (8.0 is 08:00); the model has a circadian clock |
| `meals` | list of `{"t_min", "carbs_g", "fat_g", "protein_g"}`; `[]` for none. An oral glucose load is carbs only. |
| `assumed` | the keys of this protocol whose values **the paper does not state**; `[]` if it states them all. Below. |
| `prefast_hours` | optional: hours already fasted when the arm starts (overnight is about 10) |
| `init` | optional: `cold` (default; the person is spun up through a normal fed day, then fasted `prefast_hours`) or `norm_center` (start from typical values; use when the paper's subjects were not in a clean fasted state) |
| `population` | optional override, above |

**`assumed`.** A paper rarely states everything a simulation needs. "In the
morning after an overnight fast" gives neither a clock hour nor a number of
hours, and a fasting sample has no duration at all. You still have to enter a
value: use 8.0 for a morning and 10 for an overnight fast, and a short run-in
for a single sample. Then name every key you filled in that way:

```json
"start_hour": 8.0, "prefast_hours": 10, "assumed": ["start_hour", "prefast_hours"]
```

A key is stated when the paper gives its value in words or numbers ("a 10-hour
fast", "at 08:00", "samples were drawn for 240 min"). Anything else is assumed,
however reasonable: a lower bound ("at least 12 h") is not a value. The second
reader checks this list against the paper as closely as the numbers, and the
gate prints it.

Do not assume a meal, a dose or an intensity: if the paper does not report what
the subjects ate or were given, the protocol cannot be written (leave those
findings out, or refuse with kind `input`). What you may do is **derive** an
input from numbers the paper prints, by the arithmetic below and no other. Then
the key goes in `assumed`, and the protocol's `description` gives the arithmetic
with the printed numbers it used, so that the second reader can redo it.

* **A meal given as energy and percentages of energy**: grams are
  `kcal x fraction / 4` for carbohydrate and protein and `/ 9` for fat.
* **A meal or a load given per kilogram**: times the group's mean body weight, if
  the paper prints it.
* **Exercise.** `activity` is the model's own scale, defined by oxygen uptake:
  `level = (VO2 - 3.7) / 31.7`, with VO2 in mL/kg/min. (Rest is 3.7; level 0.7
  is 25.9 mL/kg/min, which is 130 W on a cycle ergometer for a 70 kg man.) Use
  the oxygen uptake the paper reports for the bout: measured, or a stated
  percentage of the group's measured maximum times that maximum, or METs
  times 3.5. For a cycle ergometer work rate with no oxygen uptake,
  `level = watts / 186`. "70% of VO2max" with no VO2max printed, a heart-rate
  target, or a perceived exertion is not an intensity the model can be given:
  leave the bout out.

A protocol holds what its findings need and no more. For a sample drawn before
the first meal of a study day, end the arm after the sample: do not enter the
day's meals and sleep for a number that does not depend on them. It follows
that a fasting value needs no meal: when the paper's meal cannot be written (no
grams, no body weight), its fasting values still can, in a short fasted arm.

An arm that runs through a night needs the night. Awake is the baseline, so an
arm with no `sleep_wake` is a person who never slept. Enter the hours in bed
the paper gives; if it gives none, enter 23:00 to 07:00 (the night of the
model's own ordinary day) and put `sleep_wake` in `assumed`.

Optional inputs, each `{"segments": [{"from_min", "to_min", "level"}, ...]}` (and
`"unit"` where shown). A segment covers the whole minutes `from_min <= t <
to_min`: an input that must still be on for a sample at minute 120 runs to 121.
Outside the segments an input is at its baseline.

| input | baseline | level |
|---|---|---|
| `sleep_wake` | 1 (awake) | 0 asleep to 1 awake |
| `activity` | 0 (rest) | 0 to 1, intensity (0.7 is a hard cycling bout) |
| `glucose_infusion` | 0 | `unit`: `mg/dL/min` or `mg/kg/min` |
| `glucose_clamp` | 0 | target plasma glucose; `unit` is a glucose unit; 0 means no clamp that minute |
| `insulin_clamp` | 0 | plasma insulin plateau; `unit` is an insulin unit |
| `sglt2` | 0 | 1 is the steady state of empagliflozin 25 mg/day |
| `bile_fistula` | 0 | 1 drains bile completely |
| `glucose_bolus` | | `{"dose_g_per_kg", "start_min", "over_min"}`: an IV glucose bolus, as the IVGTT is run |

**A clamp** is entered at the plasma levels it held, because that is what
`insulin_clamp` and `glucose_clamp` are. Use the insulin and the glucose the
paper measured during the clamp, in the group and the arm you enter: then they
are stated. For glucose, use the paper's target if it prints no measurement. A
level taken from an approximate or pooled figure ("about 90 µU/mL", a mean over
both study conditions) goes in `assumed`. An insulin infusion rate
(40 mU/m²/min) is not a plasma level: with no measured insulin the clamp cannot
be entered (`input`, `insulin_infusion_rate`). What you enter as an input is
not also a finding.

If the paper's protocol needs an input that is not in this list (a drug, an
infusion of another hormone, hypoxia, cold, a posture change), write a refusal
of kind `input`. Do not approximate it with something else.

### `findings`

```json
{"id": "ogtt_glucose_peak", "description": "...", "marker": "glucose",
 "statistic": "peak_value", "protocols": ["ogtt"],
 "window": {"start_min": 0, "end_min": 120},
 "value": 8.0, "unit": "mmol/L",
 "uncertainty": {"kind": "sd", "value": 1.2, "n": 12},
 "location": {"where": "Table 2, row 'peak'", "quote": "Peak glucose was 8.0 (SD 1.2) mmol/L"}}
```

* `id` is unique in the file; the model sees it as `<paper>.<id>`.
* `marker` is a name from the marker table below.
* `value`, `unit`: the number **as the paper prints it, in the paper's unit.** The
  loader converts. Do not convert by hand; do not round.
* `location.where` names the table, figure or section; `location.quote` is the
  reported text with the number in it, copied. A second reader will check every
  number against this quote. For a table, quote the row's label and the cell,
  separated by ` | ` (`"Glucose, mg/dL | 85.1 ± 2.8"`), and name the column in
  `where`.
* `window` is minutes from the start of the arm, `start_min <= t < end_min`, and must
  lie inside `duration_min`.
* An optional `unusual_value_ok: "<reason>"` waives the check that a level lies in a
  plausible range for its marker (a value outside it is almost always a unit slip).
  Use it only when the paper really reports such a value.
* An optional `acknowledged_conflicts: ["<paper>.<id>"]` is explained under the gate.

### `left_out`

Almost every paper reports more than the model can take. Each thing you leave
out is recorded with the piece that was missing, so that the missing pieces can
be counted over all papers (`python -m pulse.admit --missing`): the count is
what the model and this format should learn next.

```json
{"what": "glucose incremental area 0-120 min, by fast length (Tables 1 and 2)",
 "missing": {"kind": "operator", "name": "single_arm_incremental_auc"}}
```

* `what`: one line, with where it is in the paper. One entry for a family of
  numbers (all the C-peptide values), not one per number. No values: this is a
  list of gaps, not of findings.
* `missing.kind` is one of the refusal kinds (`marker`, `input`, `population`,
  `timescale`, `operator`, or `source` when the gap is in the paper and not in
  the model; see "When to write a refusal instead").
* `missing.name` is lowercase with underscores. **Reuse a name that is already
  in use** (run `--missing` to see them) before making one: two names for one
  gap are counted as two gaps. Names in use for `source`: `figure_only`,
  `unlabelled_uncertainty`, `uncertainty_not_printed`, `zero_uncertainty`,
  `unit_not_printed`, `unit_implausible`, `value_not_printed`,
  `time_not_printed`, `input_not_printed` (a meal, a dose, an infusion or an
  exercise intensity the paper does not give), `table_ambiguous`,
  `assay_not_comparable`.
* Leave out of `left_out` what nobody would enter: numbers about markers far
  from physiology the model covers (liver enzymes, questionnaire scores), a
  duplicate of a number you did enter, and statistics of the authors' own
  model fits.

#### The statistics

| `statistic` | the paper reports | protocols | `value` | uncertainty kinds |
|---|---|---|---|---|
| `mean_in_window` | the mean of the marker over the window | 1 | a level | `sd`, `sem`, `ci`, `range` |
| `peak_value` | the highest value in the window | 1 | a level | `sd`, `sem`, `ci`, `range` |
| `time_to_peak` | when the peak occurred | 1 | a time (`min`, `h`, `s`) on the **paper's clock from the start of the arm** | `sd`, `sem`, `ci`, `range` |
| `delta_means` | mean(second protocol) minus mean(first), over the window | 2, in the order *first*, *second* | a difference | `sd`, `sem`, `ci`, `range` |
| `delta_peaks` | peak(second) minus peak(first) | 2 | a difference | `sd`, `sem`, `ci`, `range` |
| `ratio_means` | mean(second) divided by mean(first) | 2 | unit `ratio` | `sem`, `ci` |
| `ratio_increments` | area above each arm's own baseline, second over first | 2 | unit `ratio` | `sem`, `ci` |
| `spread` | the between-person standard deviation of the person's window mean | 1 | an SD, in the marker's unit | `n_only`, `sem`, `ci` |
| `correlation` | Pearson r, across people, of the window mean with `partner` | 1 | unit `r` | `n_only`, `sem`, `ci` |
| `curve` | the marker at several timepoints | 1 | per point | per point: `sd`, `sem`, `ci`, `range` |

* Every statistic is taken **per person and then averaged over people**. So
  `peak_value` is the mean of each subject's own highest value (what papers call
  the peak or Cmax), wherever in the window it fell. The highest point of the
  mean curve is a different and smaller number: enter the mean curve as a `curve`.
* `ratio_increments` also takes `baseline_window` (the run-in, ending before `window`
  starts). It is for "incremental area under the curve" comparisons, where the fasting
  level must cancel.
* `correlation` takes `partner`: another marker's name, or one of the person factors
  `insulin_resistance`, `beta_cell`, `adiposity`, `fitness`, `chronotype`,
  `hpa_reactivity`, `gastric_emptying`, `sex`, `age`.
* `delta_*` and `ratio_*` order matters: `protocols: ["fasting", "ogtt"]` is OGTT minus
  fasting.
* `curve` replaces `window`, `value` and `uncertainty` with `points`: a list of
  `{"t_min", "value", "uncertainty"}`; `unit` is shared. Times are minutes from the
  start of the arm and must fall inside it.
* If the paper reports a statistic that is not in this table (a median with an
  interquartile range, a slope, a threshold crossing, an area under the curve of a
  single arm), that is a refusal of kind `operator`.

## Units

Write each number in the paper's unit, spelled the way the paper spells it. The
loader converts to the model's unit through the table below and nowhere else.
Case, spaces and the micro sign are not significant (`µU/mL`, `uU/mL`, `uu/ml` are one unit).
A factor raised to minus one is read as a division (`mmol L-1` is `mmol/L`), `M` as
mol/L (`pM` is `pmol/L`), and the order of the divisors does not count
(`µmol/min/kg` is `µmol/kg/min`).
**A unit that is not in the table is refused, never guessed.** If the paper
uses one (insulin in ng/mL, VO2 in mL/kg/min, glucose infusion per kg of
fat-free mass), the number cannot be converted without knowing something the
loader does not: write a refusal of kind `input` or `marker` naming the unit, or
find the paper's own conversion and say so in the quote. Do not convert by hand
and enter the model's unit unless the paper itself states the converted number.

The gate also refuses a level (a mean, a peak, a curve point) that lies outside a
plausible range for its marker, and a difference or spread wider than that whole
range. That is how the commonest slip is caught: 5.5 read as mg/dL when it was
mmol/L.

Differences, spreads and uncertainties convert by the factor alone; levels by the
factor and, for temperature only, an offset (98.6 degF is 37 degC; a 1.8 degF
difference is 1 degC).

Molar conversions use the molecular weights in `pulse/knowledge/corpus_units.py`.
Insulin uses 6.0 pmol/L per uU/mL (SI).

| marker | name | model unit | plausible level | accepted units (multiply by) |
|---|---|---|---|---|
| `glucose` | Blood Glucose | mg/dL | 40 to 400 | mg/dl (x1), mg/100ml (x1), mmol/l (x18.016), g/l (x100), mg/l (x0.1) |
| `insulin` | Insulin | μU/mL | 0 to 200 | uu/ml (x1), uiu/ml (x1), miu/l (x1), mu/l (x1), pmol/l (x0.166667), nmol/l (x166.667) |
| `glucagon` | Glucagon | pg/mL | 0 to 500 | pg/ml (x1), ng/l (x1), pmol/l (x3.4828) |
| `ffa` | Free Fatty Acids | mmol/L | 0 to 3 | mmol/l (x1), meq/l (x1), umol/l (x0.001), ueq/l (x0.001) |
| `bhb` | β-Hydroxybutyrate | mmol/L | 0 to 10 | mmol/l (x1), umol/l (x0.001), mg/dl (x0.0960615) |
| `lactate` | Lactate | mmol/L | 0 to 20 | mmol/l (x1), umol/l (x0.001), mg/dl (x0.111012) |
| `hepatic_output` | Hepatic glucose output | mg/kg/min | 0 to 12 | mg/kg/min (x1), umol/kg/min (x0.18016), mmol/kg/min (x180.16) |
| `ghrelin` | Ghrelin | pg/mL | 20 to 300 | pg/ml (x1), ng/l (x1), ng/ml (x1000), pmol/l (x3.37) |
| `leptin` | Leptin | ng/mL | 1 to 50 | ng/ml (x1), ug/l (x1), pg/ml (x0.001), pmol/l (x0.016) |
| `glp1` | GLP-1 | pmol/L | 2 to 100 | pmol/l (x1), pg/ml (x0.303242), ng/l (x0.303242) |
| `cortisol` | Cortisol | μg/dL | 1 to 40 | ug/dl (x1), ug/l (x0.1), ng/ml (x0.1), nmol/l (x0.036246) |
| `acth` | ACTH | pg/mL | 5 to 120 | pg/ml (x1), ng/l (x1), pmol/l (x4.541) |
| `hr` | Heart Rate | bpm | 30 to 220 | bpm (x1), beats/min (x1), /min (x1), 1/min (x1) |
| `hrv` | HRV (RMSSD) | ms | 0 to 300 | ms (x1), s (x1000) |
| `sbp` | Systolic BP | mmHg | 60 to 250 | mmhg (x1), kpa (x7.50062) |
| `dbp` | Diastolic BP | mmHg | 30 to 150 | mmhg (x1), kpa (x7.50062) |
| `temp` | Core Temperature | °C | 35 to 42 | °c (x1), degc (x1), °f (x0.555556 -17.7778), degf (x0.555556 -17.7778), k (x1 -273.15) |
| `rr` | Respiratory Rate | /min | 4 to 40 | /min (x1), 1/min (x1), breaths/min (x1) |
| `spo2` | SpO₂ | % | 70 to 100 | % (x1), fraction (x100) |
| `liver_glycogen` | Liver glycogen | g | 0 to 150 | g (x1) |
| `muscle_glycogen` | Muscle glycogen | g | 0 to 600 | g (x1) |
| `mitochondrial_capacity` | Mitochondrial capacity | × pop. mean | 0 to 3 | ×pop.mean (x1), x_pop_mean (x1) |
| `crh` | CRH | pg/mL | 0 to 500 | pg/ml (x1), ng/l (x1), pmol/l (x4.758) |
| `insulin_action` | Insulin action (remote) | a.u. | -5 to 5 | a.u. (x1) |
| `cck` | Cholecystokinin | pmol/L | 0 to 30 | pmol/l (x1) |
| `gallbladder_bile` | Gallbladder bile acids | mmol | 0 to 12 | mmol (x1), umol (x0.001) |
| `intestinal_bile` | Intestinal bile acids | mmol | 0 to 10 | mmol (x1), umol (x0.001) |
| `bile_acids` | Serum total bile acids | µmol/L | 0 to 100 | umol/l (x1), nmol/ml (x1), nmol/l (x0.001), mmol/l (x1000) |
| `fat_mass` | Fat mass | kg | 4 to 60 | kg (x1), g (x0.001), lb (x0.453592) |
| `insulin_slow` | Insulin (slow) | μU/mL | 0 to 80 | uu/ml (x1), uiu/ml (x1), miu/l (x1), mu/l (x1), pmol/l (x0.166667), nmol/l (x166.667) |
| `insulin_potentiator` | Insulin potentiator | mg/dL | 0 to 200 | mg/dl (x1) |
| `sns_tone` | Sympathetic tone | fraction | 0 to 1 | fraction (x1), % (x0.01) |
| `epinephrine` | Epinephrine | pg/mL | 0 to 1500 | pg/ml (x1), ng/l (x1), ng/ml (x1000), pmol/l (x0.1832), nmol/l (x183.2) |
| `gastric_carb` | Gastric carbohydrate | g | 0 to 200 | g (x1) |
| `gastric_protein` | Gastric protein | g | 0 to 150 | g (x1) |
| `gastric_fat` | Gastric fat | g | 0 to 150 | g (x1) |
| `circ_phase_offset` | Circadian phase | h | -6 to 6 | h (x1), min (x0.0166667) |
| `sleep_debt` | Sleep debt | h | -16 to 16 | h (x1), min (x0.0166667) |
| `gip` | GIP | pmol/L | 0 to 150 | pmol/l (x1), pg/ml (x0.200662) |
| `glucose_infusion_rate` | Glucose infusion rate | mg/kg/min | 0 to 20 | mg/kg/min (x1), umol/kg/min (x0.18016), mmol/kg/min (x180.16) |
| `glucose_disposal` | Glucose disposal | mg/kg/min | 0 to 30 | mg/kg/min (x1), umol/kg/min (x0.18016), mmol/kg/min (x180.16) |
| `urinary_glucose` | Urinary glucose | mg/min | 0 to 1000 | mg/min (x1), mg/h (x0.0166667), g/h (x16.6667), g/day (x0.694444), g/24h (x0.694444), mmol/min (x180.16), mmol/day (x0.125111), mmol/24h (x0.125111) |
| `vo2` | Oxygen uptake | mL/min | 50 to 6000 | ml/min (x1), l/min (x1000) |
| `vco2` | Carbon dioxide output | mL/min | 50 to 6000 | ml/min (x1), l/min (x1000) |
| `rq` | Respiratory quotient | ratio | 0.5 to 1.5 | ratio (x1), rq (x1) |
| `energy_expenditure` | Energy expenditure | kcal/min | 0.2 to 40 | kcal/min (x1), kcal/h (x0.0166667), kcal/day (x0.000694444), kcal/24h (x0.000694444), kj/min (x0.239006), w (x0.0143403), kj/day (x0.000165976), mj/day (x0.165976) |
| `lipolysis` | Lipolysis | mmol/L/min | 0 to 5 | mmol/l/min (x1), umol/l/min (x0.001) |
| `gng_fraction` | Gluconeogenic fraction of glucose production | fraction | 0 to 1 | fraction (x1), % (x0.01) |
| `fat_absorption` | Fat absorption | fraction | 0 to 1 | fraction (x1), % (x0.01) |

The peak-time units are `min`, `h`, `s`. Ratios are `ratio`; correlations `r`.

## Stating uncertainty

Every number carries an `uncertainty` of a stated kind **and its n**. You report
what the paper printed; the loader works out how hard the model should be pulled.
Do not compute a standard error yourself and do not choose a sigma.

| `kind` | fields | when |
|---|---|---|
| `sd` | `value`, `n` | "mean (SD)" or "mean +/- SD": the spread between people |
| `sem` | `value`, `n` | "mean +/- SEM", error bars labelled standard error |
| `ci` | `low`, `high`, `level` (90, 95 or 99), `n` | a confidence interval **of the reported statistic** |
| `range` | `low`, `high`, `n` | the smallest and largest individual |
| `n_only` | `n` | only for `spread` and `correlation`, when the paper gives the statistic and its n and nothing else |

`n` is the number of **people** behind the number, not of tests or samples: twelve
subjects each tested three times are 12. If the paper says its standard error or
interval was computed over more measurements than people ("n = 36: 12 subjects x
3 repeated tests"), add `"observations": 36` beside `"n": 12`. The loader then
recovers the spread with the count the paper used and holds the mean only to what
twelve people support. If error bars are not labelled
(SD or SEM?) in text, caption or methods, do not guess: the finding cannot be
entered; leave the number out (`left_out`, kind `source`, name
`unlabelled_uncertainty`), or refuse the paper with that if nothing else is left. The statistic has to lie inside its own `ci` or `range`.

What the loader does (`pulse/knowledge/corpus_uncertainty.py`): the model scores the
mean of its virtual people against the paper's mean at a standard error
`sigma / sqrt(n_arm)`, where `sigma` is the **between-person SD** and `n_arm` the
paper's n. From an `sd` those are the SD and n. From an `sem` or a `ci` the SD is
recovered (`sem * sqrt(n)`; the t quantile for a CI; Wan et al. 2014 for a range).
For a curve, a ratio of increments, a spread or a correlation the sigma is the
standard error itself. **n is capped at 25**: a paper with 200 subjects
is held to the standard error 25 would give, because one cohort's mean does not pin
the population's mean to 7% of its spread. A sigma tighter than the study supports
can wrongly exclude the truth, and that is the one mistake the format is built to
prevent.

A one-sided claim ("below 140 mg/dL", a diagnostic threshold, "rises", "is higher
than") has no reported number with an uncertainty and does not belong in a finding.

## When to write a refusal instead

Write a refusal when the paper is about something the model cannot be told. It is
not a failure: the pile of refusals is the specification for the model's next
change, and it is counted by kind.

```json
{
  "paper": "example-2098",
  "citation": {
    "authors": "Example C",
    "title": "A made-up study of a marker the model does not carry",
    "journal": "Journal of Made-Up Examples",
    "year": 2098,
    "pmid": "99999996"
  },
  "reported": [
    {
      "what": "plasma adiponectin fell 20% after the meal",
      "location": {
        "where": "Figure 2",
        "quote": "Adiponectin fell by 20% at 120 min."
      }
    }
  ],
  "missing": {
    "kind": "marker",
    "name": "adiponectin"
  },
  "detail": "The finding is about adiponectin, which the model has no state for."
}
```

`missing.kind` is exactly one of:

| kind | the missing piece | `name` examples |
|---|---|---|
| `marker` | something measured that the model has no series for. If the name is already a marker or observable, the file is refused. | `adiponectin`, `free_testosterone` |
| `input` | something done to the subjects that the protocol cannot express, or a unit that cannot be converted | `hypoxia`, `metformin`, `growth_hormone_infusion`, `insulin_in_ng_per_ml` |
| `population` | subjects the model cannot represent | `type_2_diabetes`, `pregnancy`, `children` |
| `timescale` | a duration or resolution the model does not simulate | `weeks_of_training`, `sub_minute_pulsatility` |
| `operator` | a kind of number the statistics do not cover | `median_with_iqr`, `slope`, `single_arm_incremental_auc` |
| `source` | not a gap in the model: the paper does not give what is needed | `figure_only`, `unlabelled_uncertainty`, `input_not_printed` |

`missing.name` is lowercase with underscores; reuse a name already in use
(`python -m pulse.admit --missing`) before making one.

`reported` lists what the paper reported that you could not enter: each with
`what`, and a `location` (`where`, `quote`) as for a finding. `detail` is one
sentence saying why the model cannot hold it. Use one refusal per paper, naming
the **one** piece that, if added, would let you write the file. If a paper has some
findings the model can express and some it cannot, write the finding file for the
first and no refusal: a paper is either a finding or a refusal, never both.

`python -m pulse.admit --refusals` counts the pile by kind and by name, and
`--missing` counts it together with what the finding files left out.

## Running the gate

```
uv run python -m pulse.admit corpus/findings/<paper>.json
uv run python -m pulse.admit corpus/refusals/<paper>.json
```

Exit 0 means admitted. The gate checks, and names the check that failed:

1. **the file parses**: strict; protocols well formed; units listed and values
   plausible; every number has a place in the paper (`where` and a `quote` containing
   the number); every uncertainty is of a stated kind;
2. **the paper is filed once**: the file is named for its paper; no other file or
   Python registry holds it; no other slug carries its DOI, PMID or PMCID; it is not
   also a refusal;
3. **each protocol runs** on the untrained student;
4. **each finding's number moves when the model's parameters move**: otherwise the
   model could not move the number whatever it learned. The gate rolls every protocol
   of the file once, nudges all the parameters along two random directions, rolls
   again from the same start, and compares each finding's predicted statistic. It
   proves that the number depends on some parameter, as training would see it. It
   does not say which parameter, or that the dependence is strong enough to learn
   from;
5. **no finding silently contradicts one already admitted**: on one protocol, marker,
   statistic and window, with targets further apart than their standard errors allow.
   If the disagreement is real, name the other item in `acknowledged_conflicts` so it
   is recorded, not hidden: a real conflict is the signal to change the model's
   structure, never to widen a sigma. Acknowledging a conflict that does not exist is
   refused.

It then lists what each protocol `assumed`, and says whether the paper is
**sealed** (held out of training) or trains.

The gate uses only the untrained model and never prints how well any model fits a
finding. A finding is not better for agreeing with the model, and a paper is never to be
dropped because the model scores badly on it. Write what the paper says. The
second reader checks each number against its quote; the gate checks that it can be
used.

## Things that bite

* A meal is `{"t_min", "carbs_g", "fat_g", "protein_g"}`. Name all three.
* A 75 g OGTT is `carbs_g: 75`. Sugar in a drink is carbohydrate.
* Report the window the paper used. A "2-hour glucose" is a single time, not a mean
  of the second hour: use `mean_in_window` with a window of one minute
  (`end_min = start_min + 1`).
* IV glucose in g/kg is a `glucose_bolus`; per minute is a `glucose_infusion`.
* The clamp's glucose infusion rate is the marker `glucose_infusion_rate` (mg/kg/min).
* Fasting studies: set `prefast_hours`. A "morning after an overnight fast" is about 10.
  A sample after 72 hours without food is a short arm with `prefast_hours: 72`, not
  a three-day arm: the person is spun up through a normal fed day and then fasted
  that long.
* `glucose` is blood (plasma) glucose. A glucose sensor reads interstitial fluid:
  leave its numbers out.
* An uncertainty printed as zero (`0.1 ± 0.0`) is a rounded number, not a
  measurement without error. Leave that number out.
* `ghrelin` in the model is a nominal level that rests near 100 pg/mL. It does not
  say active or total ghrelin, whose assays differ several-fold. Leave absolute
  ghrelin levels out.
* A unit the paper does not print is not to be supplied from convention (RMSSD "in
  ms"): leave the number out.
* If the unit as printed converts to an absurd value (adrenaline in ng/mL), the paper
  has probably misprinted it. Leave the number out; do not correct the paper.
* When a table does not say when a baseline value was drawn, enter it on a fasted
  morning and put `duration_min` in `assumed`.
* A sample drawn a few hours after food the paper does not describe (subjects who
  "abstained for at least four hours") is not a fasting sample, and the state it
  was drawn in cannot be written: leave it out (`source`, `input_not_printed`).
* A "fasting" or "baseline" value that the paper defines as the mean of several
  samples (-20, -10 and 0 min) is a `mean_in_window` over the minutes those samples
  span, in a run-in before the load.
* A curve's `quote` joins its table rows with ` ; `, one row per time point, when
  the table has a row for each time.
* A number read only at minute 0 fails check 4: minute 0 is the state the protocol
  starts from, and training does not differentiate through how it got there. Give a
  baseline as the mean over a run-in (start the protocol 30 minutes before the load
  and put the meal at `t_min: 30`), or keep it as the first point of a curve, where
  the later points carry it.
* One paper, many arms: declare each arm once, and let findings share it.
* Do not copy numbers from the abstract of a paper you have not read; do not copy
  from a review.
