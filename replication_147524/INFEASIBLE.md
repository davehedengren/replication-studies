# 147524 — The Government Spending Multiplier in a Multi-Sector Economy

**Authors:** Hafedh Bouakez, Omar Rachedi, Emiliano Santoro
**Journal:** American Economic Journal (forthcoming / in press at time of package)
**Status:** INFEASIBLE for Python replication

## Paper summary

The paper builds a calibrated multi-sector New Keynesian DSGE model with 57
industries, input-output linkages from the BEA U.S. I-O matrix, sector-specific
Calvo price rigidity, capital adjustment costs, and government spending. The
headline result is that the aggregate value-added government spending
multiplier is 0.74 in the multi-sector baseline versus 0.42 in a one-sector
model — a 75 percent amplification driven primarily by I-O linkages. The ZLB
version delivers 1.98 vs 1.07. A short empirical section (Section IV, Table 4)
runs an IV regression of sectoral cumulative value added on cumulative
government spending interacted with a sector-level centrality/upstreamness
measure, using the Ramey-Zubairy (2018) fiscal news shock as an instrument, to
validate the model's sectoral prediction.

## Why this paper is infeasible for this project

### 1. The bulk of the paper is Dynare/MATLAB and cannot be ported to Python at reasonable cost

`RunMainScript_Model.m` drives ~90+ Dynare subdirectories under
`ReplicationPackage/Models/` (e.g., `Het`, `Het_ZLB`, `Het_CRRA`,
`Het_StickyWage`, `Het_InflTarg_NoIO`, `Het_rho_045_NoProd`, `1Sect_*`, ...).
Each is a standalone Dynare `.mod` file or MATLAB script that solves the
calibrated model, computes impulse responses and multipliers, and writes the
entries for the paper's Tables 1-3, 5-6 and Figures 1-4. Dynare is a MATLAB
toolbox for solving DSGE models symbolically and linearly; it has no
first-class Python equivalent. The only serious substitute (`dolo`, `gEconpy`)
would require hand-reimplementing every model-specific `.mod` file, every
sectoral calibration, and re-deriving the perturbation/perfect-foresight
solvers — effectively rewriting ~95 percent of the paper from scratch. This
matches the "MATLAB-only structural model" infeasibility criterion in the
driver instructions, and is consistent with how the driver has already handled
comparable Dynare-heavy papers (e.g., 225841 — Confidential CoreLogic/CBRE data
+ MATLAB/Dynare model; 130626 — Structural SIR optimal-control; 141001 —
Structural Fortran+MATLAB sovereign-default).

### 2. The one empirical piece (Table 4) cannot be run from the shipped package

`Data/RunGenerateData.do` imports four spreadsheets to build the Table 4 panel:

| File | Present? | Notes |
|------|----------|-------|
| `Data/RZDAT.xlsx` | **MISSING** | Ramey-Zubairy (2018) quarterly fiscal dataset — `ngov ngdp pgdp news rgdp nfedcurrreceipts_nipa rgdp_potcbo rgdp_pott6` |
| `Data/CentralityTable.xlsx` | **MISSING** | Paper-specific sectoral upstreamness/centrality measure computed from the BEA I-O matrix |
| `Data/NIPA_TABLE_3_10_5.xls` | present | Government purchases of intermediate goods |
| `Data/GDPbyInd_VA_1947-2017.xlsx` | present | BEA sectoral value added |

Two of the four inputs are not in the package. `RZDAT.xlsx` is on Valerie
Ramey's website and could in principle be downloaded, but
`CentralityTable.xlsx` is a paper-specific construct derived from the authors'
own processing of the BEA Use table (each sector's position in the production
network, after dropping sectors 27-29 and 51 per the `.do` file). Rebuilding it
faithfully would require replicating the authors' I-O processing — which is
itself embedded in the uncompiled MATLAB code — and would leave any Table 4
numbers non-comparable to the published ones anyway, since the centrality
values drive the coefficient of interest.

### 3. Tables / Figures / InTextNumbers output folders are empty

`ReplicationPackage/Tables/`, `Figures/`, and `InTextNumbers/` are all empty in
the shipped package, so there is no pre-computed intermediate output we could
re-verify against the PDF without re-running the Dynare/MATLAB pipeline.

## What a Python replication would and would not cover

Even if we downloaded `RZDAT.xlsx` from Ramey's site and hand-constructed a
centrality proxy, the most we could cover is a single IV regression (Table 4,
four columns) — roughly **3 percent** of the paper's empirical content. The
other 97 percent (Tables 1-3, 5-6, all figures, all counterfactual
decompositions, all ZLB results, all sensitivity exercises across the 90+
Dynare variants) would remain unverified. This falls below the bar for a
"partial replication" entry as used elsewhere in this repo, where partial
replications generally cover the paper's core empirical tables and only
exclude a clearly separable structural/calibration appendix.

## Decision

Skip as infeasible. The paper is a structural DSGE study whose main results
live inside Dynare `.mod` files with no Python equivalent, and the small
empirical validation section is missing its two critical input files.
