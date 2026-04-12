# Paper 233062 — Infeasible for Python Replication

**Paper**: Capelle & Liu, "Optimal Taxation of Inflation" (March 2025 draft, AER replication package ID 233062)

## Why infeasible

The replication package is a purely structural/DSGE exercise, implemented
entirely in MATLAB and Dynare. There is no empirical regression-based
component that can be ported to Python in the time budget this driver
allows.

Specifically, the package contains three sections, each a self-contained
MATLAB/Dynare project:

1. **Section 4 — Small-scale New-Keynesian model** (`Section4_Rep_Model/`)
   - Main driver `Section4_Run.m`, calibrated NK model simulations
     (`A1_main_model.m`, `A2_lower_stickiness.m`, `A3_wage_subsidy.m`), and
     two Dynare `.mod` files (`Dy_TIP_basic.mod`, `Dy_TIP_subsidy.mod`).
   - Produces the stabilization-gain numbers cited in the introduction
     ("inflation variance down 45%, output down 44%" for a calibrated
     markup shock). These are moments of simulated IRFs, not estimates
     from data.

2. **Section 6 — Multi-sector relative-price model** (`Section6_Rel_Price/`)
   - Two-step calibration via SMM (`A1_Calibrate_Step1_InitialGuess.m`,
     `A2_Calibrate_Step2_MultiSector.m`), steady-state-Jacobian solver
     (`B1_Compute_SSJ.m`), nonlinear MIT-shock IRFs
     (`B2_Compute_Nonlinear_IRF.m`), and figure generation.
   - Relies on the Sequence-Space Jacobian (SSJ) toolbox, a model-solution
     library with no mature Python port suitable for this codebase.
   - Input moments come from PCE micro-data preprocessed in Stata
     (`data_cali/pce_data/compute_moments_HP.do`) and loaded from an
     `.xlsx` weighting file; only the final weight matrix and moment
     vector are shipped, not the raw PCE extract.

3. **Section 7 — Smets-Wouters (2007) DSGE with TIP extension**
   (`Section7_DSGE/`)
   - Full Dynare estimation (`SW2007_estimate.mod`, `SW2007_TIP.mod`) with
     four counterfactual mode files (`..._ffer.mat`, `..._gs3.mat`,
     `..._ssr.mat`, `..._wx_sr.mat`) and a large pre-saved posterior mode
     (`SW2007_mode_tmp.mat`).
   - The estimation is a Bayesian MCMC over a medium-scale DSGE; it
     requires Dynare, which has no Python equivalent (the closest,
     `dolo`/`dolark`, does not handle models of this size and neither does
     `estimagic` for this specific likelihood structure).

## File-level evidence

```
$ find 233062-V1/Rep_Package_20250612 -type f \( -name "*.m" -o -name "*.mod" \) | wc -l
     159
$ find 233062-V1/Rep_Package_20250612 -type f \( -name "*.py" -o -name "*.R" -o -name "*.ipynb" \) | wc -l
       0
```

The only non-MATLAB analytical code in the whole package is
`Section6_Rel_Price/data_cali/pce_data/compute_moments_HP.do`, a Stata
script that HP-filters PCE sector data to produce the moments the SMM
calibration targets. Running that alone would not replicate any result in
the paper — it is only a preprocessing step for an untouchable downstream
MATLAB calibration.

## What a real replication would require

- A licensed MATLAB install with Dynare 5.x+.
- The SSJ (Auclert, Bardóczy, Rognlie, Straub) toolbox.
- Several hours of solver + MCMC compute for Section 7.
- Manual porting of thousands of lines of MATLAB/Dynare to Python, which
  is a research project in its own right — not a replication.

None of this fits the Python-only, single-session replication workflow
that the rest of this repo is built around. Marking as infeasible is the
honest call.

## What I did not do

- No scripts in `replication_233062/` beyond this file.
- No attempt to re-run any `.m` or `.mod` file — MATLAB/Dynare are not
  available in the shared environment.
- No import of partial outputs into Python just to generate a plot.

## Classification

Skipped — **MATLAB/Dynare-only structural DSGE paper**. Analogous to other
skips in this repo's Skipped Papers table where the replication package
has no Python-reachable empirical layer.
