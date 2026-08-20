# The biliary axis — literature anchors and model design

Everything the iter-95 hepatobiliary states are built from, with sources. Written before
the code so the constants have a contract to answer to rather than being chosen to make a
curve look right. Numbers here were looked up for this iteration, **not** recalled — the
iter-94 circadian work is the precedent for what happens when an encoded constant and the
range it cites drift apart.

Design rationale for entering the liver axis here rather than at ALP/GGT/ALT is in
`docs/iter95-proposal.md` §3.

---

## 1. Anchors

### 1.1 Cholecystokinin

| quantity | value | note |
|---|---|---|
| fasting plasma CCK | **0.8–1.2 pmol/L** (mean ~0.9) | |
| postprandial peak | **6.5–7.1 pmol/L** | mixed liquid meal |
| time to peak | **~10 min** | fast — this is the *stimulus*, not the readout |
| level at 30 min | **~3.5 pmol/L** | falls quickly off peak |
| duration elevated | **3–5 h** | while food empties from the stomach |

Secreted by duodenal I-cells in response to intraluminal **fat and protein**, which is why
the gut module's existing lipid and amino appearance channels are the right stimulus and
no new model input is required.

### 1.2 Gallbladder

| quantity | value | note |
|---|---|---|
| fasting volume | **10–40 mL**, most 10–25 mL | sources disagree; higher figures (50–70 mL) appear to be *capacity*, not fasting volume |
| ejection fraction | **≥35–38 % is normal**, measured at 60 min | the HIDA/CCK clinical threshold |
| emptying shape | **biphasic** — early rapid, late slow | still emptying at 60 min in most subjects, which is *why* GBEF is quoted at 60 min |

### 1.3 Serum total bile acids

| quantity | value | note |
|---|---|---|
| fasting reference interval | **4.4–14.1 µmol/L** | |
| postprandial reference interval | **4.7–20.2 µmol/L** | |
| a trial's healthy means | fasting **8.6**, postprandial peak **11.9 µmol/L** | modest fold-rise in the mean; the interval's upper bound moves much more |
| rise begins | **within 30 min** of the meal | |
| time to peak | **75–120 min** | |

**The 10 min → 75-120 min gap between the CCK peak and the serum bile-acid peak is the
single most important feature to reproduce.** It is not a fitted lag: it is gallbladder
emptying, intestinal transit, ileal reabsorption and hepatic first-pass extraction
happening in series. A model that hits it by tuning a delay constant has not earned it.

### 1.4 Enterohepatic circulation

| quantity | value |
|---|---|
| total bile-acid pool | **~3 g** |
| recycles | **4–12 × per day** |
| ileal reabsorption efficiency | **~95 %** |
| faecal loss | **~5 %**, replaced by hepatic synthesis |

---

## 2. Model design

### 2.1 Four states, because the delay must be earned

```
  meal lipid/protein ──> [CCK] ──gate──> [GALLBLADDER] ──emptying──> [INTESTINE]
                                              ^                            │
                                              │                    ileal reabsorption (~95%)
                                     canalicular export                    │
                                              │                            v
                                        [hepatocyte] <──── portal return ──┘
                                              │
                                       first-pass extraction
                                              │
                                     spillover (small) ──> [SERUM BILE ACIDS]
```

| state | unit | typical | why it is a state |
|---|---|---|---|
| `cck` | pmol/L | 1.0 | the driver; secretion is meal-gated, clearance is fast |
| `gallbladder_bile` | mmol | ~4 | a **pool** — it depletes, and cannot be emptied twice |
| `intestinal_bile` | mmol | small | carries the transit delay AND the 95 % / 5 % split |
| `bile_acids` | µmol/L | ~6 | the observable |

`intestinal_bile` is the state I was initially reluctant to add (it makes the axis four
markers, not three). It earns its place twice over: it is where the CCK-peak-to-serum-peak
delay *comes from* rather than being fitted, and it is where reabsorption efficiency lives,
so the 95 %/5 % split is a mass balance rather than a constant.

### 2.2 Contraction is a gate, not a state

Settled in `docs/iter95-proposal.md` §3.2.1 and repeated here because it is the easiest
thing to get wrong: the gallbladder is the axis's one genuine **pool**, and contraction is
a dimensionless fraction fully determined by CCK at each instant. Making contraction the
state and holding volume constant would mean a second meal 90 min after the first produces
an identical excursion, because the reservoir never depletes. Ejection fraction is
recovered as a **derived** quantity (ΔV/V₀ over 60 min) and scored against §1.2.

### 2.3 The canalicular export step is explicit — this is the point

Cholestasis *is* failure of canalicular export. Modelling it as an explicit rate constant
(`k_canalicular`) on the hepatocyte→bile flux means that later, ALP/GGT/ALT/bilirubin hang
off *impairment of a step that already exists*, rather than needing a driver bolted on. If
export capacity falls, hepatic first-pass extraction saturates, more of the portal return
spills into serum, and serum bile acids rise — which is the actual clinical picture, and it
falls out of the mass balance instead of being asserted.

This is the whole reason for entering the liver axis at the mediator rather than at the
enzymes.

### 2.4 What is deliberately NOT modelled yet

- **Bilirubin.** A standard-panel observable, but with no obstruction or haemolysis
  represented it would ship inert — this project's most repeated failure mode.
- **FGF19 ileal feedback** on bile-acid synthesis. Real, and the natural next state, but it
  closes a slow loop that nothing currently scores.
- **Bile-acid species** (cholic vs chenodeoxycholic, conjugated vs free). One lumped pool
  until something requires the distinction.
- **Sphincter of Oddi tone**, gastric emptying coupling beyond the existing gut kernel.

---

## Sources

- [Fasting and postprandial serum bile acid concentrations in normal persons using an improved GLC method (Digestion, 1978)](https://pubmed.ncbi.nlm.nih.gov/627320/)
- [Intrahepatic cholestasis of pregnancy — time to redefine the reference range of total serum bile acids (2022)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9543426/) — fasting 4.4–14.1, postprandial 4.7–20.2 µmol/L
- [Determinants of fasting and postprandial serum bile acid levels in healthy man](https://pubmed.ncbi.nlm.nih.gov/677089/)
- [Comparison of fatty meal and intravenous cholecystokinin infusion for gallbladder ejection fraction (J Nucl Med, 2002)](https://jnm.snmjournals.org/content/43/12/1603)
- [Measurement of gallbladder ejection fraction (J Nucl Med, 2003)](https://jnm.snmjournals.org/content/44/9/1544)
- [Physiological concentrations of cholecystokinin stimulate amino acid-induced insulin release in humans (JCEM, 1987)](https://pubmed.ncbi.nlm.nih.gov/3305550/)
- [Regulation of gastric emptying in humans by cholecystokinin (JCI)](https://www.jci.org/articles/view/112401)
- [Cholecystokinin: clinical aspects of the new biology (Rehfeld, J Intern Med, 2025)](https://onlinelibrary.wiley.com/doi/full/10.1111/joim.20110)
- [Dynamics of the enterohepatic circulation of bile acids in healthy humans (AJP-GI, 2021)](https://journals.physiology.org/doi/full/10.1152/ajpgi.00476.2020)
- [Bile acid physiology (Annals of Hepatology)](https://www.elsevier.es/en-revista-annals-hepatology-16-articulo-bile-acid-physiology-S1665268119310385)
- [Enterohepatic circulation of bile acids — Medical Physiology, 3rd ed.](https://doctorlib.org/physiology/medical/256.html)
