# Paper 130626: INFEASIBLE FOR REPLICATION

**Paper**: "Optimal Targeted Lockdowns in a Multi-Group SIR Model"
**Authors**: Daron Acemoglu, Victor Chernozhukov, Iván Werning, Michael D. Whinston
**Outlet**: AER: Insights (2021), NBER WP 27102

**Reason**: Pure structural optimal-control simulation with no empirical component.

## What the paper does

The paper develops a multi-group (young / middle-aged / old) SIR model and
solves an optimal-control problem for differentiated lockdown policies. All
figures in the paper (Figures 3, 4, 5, A1–A8) are *simulation outputs* from
numerically solving a dynamic optimization problem — there is no dataset, no
regression table, no standard errors to reproduce.

Parameters (infection, hospitalization, and fatality rates by age group) are
calibrated from epidemiological literature (Ferguson et al. 2020 / Imperial
College; South Korea and Diamond Princess mortality; Acemoglu et al. 2020a).
These are ~20 hand-coded numbers in the notebooks, not a dataset to audit.

## Why it's out of scope

The replication framework used in this project is designed for empirical
papers with data, sample construction, and regression / descriptive outputs
that can be compared to published values. This paper has none of those:

1. **No data to clean.** The repository contains four Jupyter notebooks and a
   Readme.txt — zero data files. All parameters are literal numbers in code.
2. **No tables to reproduce.** Outputs are Pareto frontier curves, time paths
   of infections/lockdowns/deaths, and comparative-statics figures, not
   estimated coefficients.
3. **No empirical identification** to audit, placebo-test, or check for
   robustness in the sense used by the other papers in this collection.
4. **Solver-bound structural model.** The notebooks rely on the `gekko`
   optimization suite (IPOPT-based nonlinear programming) with hundreds to
   thousands of time-discretized decision variables. A Python port would be
   a line-for-line rewrite of an already-Python pipeline with no value added,
   and the robustness / audit phases of the standard workflow do not apply.

This is analogous to paper 225841 (Confidential CoreLogic/CBRE data +
MATLAB/Dynare model), which was also skipped for the structural-model
criterion, and to the structural portion of 127341 (Fajgelbaum et al.,
Optimal Lockdown in a Commuting Network), where only the reduced-form
gravity step was replicable and the SEIR/Hamiltonian pipeline was skipped.
Unlike 127341, however, 130626 has no reduced-form step at all — the entire
paper is structural.

## What exists in the package

```
130626-V1/
  Readme.txt
  Optimal3GPolicy_v6.ipynb            # Figures 3, 4, 5, A1, A3, A4, A5.2
  Optimal3G_SEIR_v6.ipynb             # Figures A6, A8.3
  Optimal4GPolicy_OldWorking_v6.ipynb # Figure A2.4
  Optimal3G_CustomContactMatrix.ipynb # Figure A7
```

All four notebooks are Google Colab notebooks that `pip install gekko`, mount
Google Drive, and solve the optimal-control problem. No CSVs, no parquet, no
.dta, no .RData.

## Verdict

Skipped. The paper's contribution (characterizing optimal targeted lockdowns
in a multi-group SIR model) is entirely theoretical / computational and
doesn't fit the replicate-an-empirical-result workflow this project is built
around. Running the notebooks would reproduce the authors' own figures from
their own code and would not provide the independent verification that is
the point of the project.
