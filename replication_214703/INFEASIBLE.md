# 214703 — TFPR: Dispersion and Cyclicality (Cooper & Ozturk, AEJ: Macroeconomics)

**Status:** Infeasible — pure MATLAB structural model with no empirical component.

## Paper

Russell Cooper (Liaoning U. / EUI) and Ozgen Ozturk (Bank of England / Oxford),
"TFPR: Dispersion and Cyclicality," forthcoming in *AEJ: Macroeconomics*.
The paper builds a menu-cost OLG model with monopolistic competition under
either a CES or Kimball aggregator and asks which combination of aggregate
shocks (monetary, aggregate TFP, idiosyncratic-dispersion) reproduces the
observed cyclicality and dispersion of measured revenue productivity (TFPR).

## Why this is not replicable in Python

The replication package is entirely MATLAB. The authors state this explicitly
in `README.txt` §1: *"Data: The analysis is based on simulated data."* No
microdata, no reduced-form estimation, no Stata/R scripts. Every number in
every table and figure is produced by simulating the structural model under
one of eight calibrations (CES Baseline / CES Macro / Non-CES Micro /
Non-CES Macro × individual-shock / tandem-shock variants).

Concretely:

- **~43 MATLAB files per calibration sub-directory**, across 8 calibration
  directories (`CES/{Baseline, Baseline Combined, Macro, Macro Combined}` and
  `Non-CES/{Micro, Micro Combined, Macro, Macro Combined}`) — roughly 300+
  `.m` files in total. Core routines include a rouwenhorst discretiser,
  value-function iteration over a (K × z × F × α × m) state space
  (`pricing.m`, `pricing_opt.m`), simulated long panels of firms
  (`simulate_burn.m`, `simulate_economy.m`), and decomposition of TFPR
  dispersion / pricing frequency into contemporaneous vs expected components
  (`tfpr_tfpq.m`, `timeregressor.m`).
- **Runtime:** authors report ~25 min (tables) + ~28 min (figures) on an
  Apple M4 just for the pre-packaged master scripts, on top of which a
  faithful Python port would need to re-implement value-function iteration,
  menu-cost fixed-point solvers, and Kimball-aggregator price-index inversion
  — weeks of work, not the 1–2 day budget of this pipeline.
- **No Python equivalent for the core estimator.** The "estimator" is
  solving and simulating a structural DSGE-style menu-cost model. There is
  no off-the-shelf Python package that does this; the nearest equivalent
  would be a Dynare or custom VFI port, which by rule is out of scope for
  this project (cf. 147524, 184261, 141001, 130626, 225841 in the skipped
  list, all of which were skipped for the same reason).
- **No empirical piece to split off.** Table 5 contains a single row labeled
  `DATA` with six moments (Contr_dispR = 0.219, Exp_dispR = 0.051, Contr_dispP
  = 0.090, Exp_dispP = 0.073, Contr_freqP = 0.161, Exp_freqP = 0.149) that the
  model is calibrated against. These are hard-coded inside `RUN_TABLES.m`
  (lines 104–105) as literal constants — they are taken from prior empirical
  work and are not computed from microdata in this package. There is no
  "reduced-form first half" that could be ported in isolation.

## What a replication would require

Implementing a faithful Python version of the paper would mean porting, at
minimum: rouwenhorst AR(1) discretisation on idiosyncratic productivity z and
demand α; the stochastic menu-cost draw F; the Bellman operator for firms
facing a two-period nominal price contract under CES (and separately under
Kimball) demand; the aggregate consistency condition pinning down the
price-index update under the money-growth rule; long-panel simulation across
dispQ, X, MUQ, and tandem shock scenarios; and the decomposition that splits
realised TFPR-dispersion changes into contemporaneous and expected components
for the β regressions in Tables 5, 8, 9. Even a rough port is a multi-week
undertaking and would not add empirical content to the paper's claims, since
every published number is a model moment.

## Decision

Skip, per the project rule: "MATLAB-only structural model with no empirical
component." Classifying as **Skipped — structural model, no empirical
component** alongside 130626, 141001, 147524, 184261, and 225841.
