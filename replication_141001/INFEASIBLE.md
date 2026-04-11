# Replication 141001 — INFEASIBLE

**Paper:** Hatchondo, Martinez & Roch, "Fiscal Rules and the Sovereign Default Premium"
**Journal:** American Economic Journal: Macroeconomics (forthcoming)
**Package:** `141001-V1/` (openICPSR 141001)

## Why this paper is out of scope

This is a pure quantitative structural macro model with no empirical estimation component. The entire "replication" pipeline is:

1. **Fortran solver** (`aej_code_*.f90`) that iterates on value, policy and bond-price functions for a sovereign-default model à la Eaton-Gersovitz. The authors report ~13 hours of runtime per economy on a modern laptop, and the package ships **13 distinct economies** (benchmark, 3 debt-brake rules, 3 spread-brake rules, 4 cost-of-default variants, 4 exclusion-parameter variants, 4 recovery-parameter variants, plus a `cons_increase` grid for welfare calculations and a `myopia` correction). A full recomputation would take roughly 150-200 hours of Fortran compute.
2. **MATLAB post-processing** (`figures_tables_aej.m`, `simulate.m`, `welfare_gains_*.m`, `figures_sim.m`) that consumes a pre-built `workspace_replication_aej.mat` binary and emits the figures and tables. All tables and figures in the paper are generated from this MATLAB workspace.
3. **No observational data.** The only "datasets" in `Data/` are:
   - `Figure1.xlsx` — a small hand-compiled series of IMF Fiscal Rules / Fiscal Council counts used for a single descriptive figure. No regression.
   - `Table2.xlsx` — Spain calibration targets (6 business-cycle moments) used as inputs to the structural calibration. No regression.

There is nothing to estimate in Python. The quantitative content of the paper is the model solution itself, and that is produced by Fortran executables from pre-specified parameter files (`calibration.txt`, `beta.txt`, `debt_limit.txt`, etc.). MATLAB then reads the Fortran output and generates the tables.

This fits the driver's explicit "MATLAB-only structural model with no empirical component" infeasibility criterion, and is in fact worse: the core model is Fortran, and MATLAB is just the reporting layer.

## What would be required to replicate

- A Fortran compiler (the shipped `.exe` files are Windows-only; the `.f90` source would need to be rebuilt on macOS/Linux with gfortran), plus ~150-200 CPU-hours to re-solve all 13 economies.
- A MATLAB license to execute `figures_tables_aej.m` against the supplied `workspace_replication_aej.mat`, or a full Python port of ~2000 lines of MATLAB covering HP-filtering, simulation statistics, welfare-gain integrals, IRF construction, and commitment-cost computations.
- Reverse-engineering the format of the ~30 intermediate `.txt` outputs per economy (policy functions on `b_grid × y_grid × chi_grid`, bond price menus, coefficients of expected value functions, etc.) to read them into numpy without the MATLAB workspace.

None of this produces an **empirical** claim that can be audited. The paper's findings ("a spread-brake of 3% would reduce the mean debt-to-GDP ratio from X to Y and raise welfare by λ% of consumption") are functions of the model's parameters and solution algorithm, not of any observable data we could re-estimate.

## Closest analogue already in the repo

- `replication_130626/` — also skipped as a structural SIR optimal-control model with no empirical component.
- `replication_120281/`, `replication_127341/`, `replication_136342/`, `replication_140141/` — partial replications where the empirical / measurement layer was reproduced in Python but the MATLAB structural layer was left out of scope.

141001 has no empirical / measurement layer at all, so there is no partial replication to do. Marking as **Skipped — structural Fortran+MATLAB model with no empirical component**.
