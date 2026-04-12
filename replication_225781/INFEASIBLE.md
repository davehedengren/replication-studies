# Infeasible: Paper 225781

**Paper:** Bolhuis, Rachapalli, and Restuccia — "Misallocation in Indian Agriculture" (AEJ: Macroeconomics)

## Reason: Raw microdata not bundled; paper is a structural calibration model

### 1. Required microdata is gated behind ICPSR registration

The replication package's `raw/IHDS-I/`, `raw/IHDS-II/`, and `raw/Merge/` folders
are **empty**. Per `README.pdf`, the replicator must separately download from
ICPSR (after agreeing to terms and registering):

- **IHDS-I (2004-05)** — ICPSR study 22626: DS1 Individual, DS2 Household,
  DS7 Village, DS8 Crops (save as `Individual.dta`, `Household.dta`,
  `Village.dta`, `Crops.dta` in `raw/IHDS-I/`)
- **IHDS-II (2011-12)** — ICPSR study 36151: DS1 Individual, DS2 Household,
  DS12 Village (save as `Individual.dta`, `Household.dta`, `Village.dta`
  in `raw/IHDS-II/`)
- **IHDS Household Linking Variables** (text file from
  https://ihds.umd.edu/data/data-download) saved as `linkhh.txt` in `raw/Merge/`

The entire pipeline — `clean_ihds1.do`, `clean_ihds2.do`, `ihds_panel.do`,
all TFP estimation, calibration, and counterfactuals — starts from these
files. Without them, no intermediate `data/Clean/` files can be produced
and nothing downstream can run. Only three small miscellaneous CSVs ship in
the package (CPI, state CPI for agri workers, Deininger-Jin-Nagarajan land
reforms) — none of which are inputs to the headline results.

The automated driver does not have authenticated ICPSR access, cannot agree
to terms of use on behalf of a user, and cannot download the IHDS microdata.

### 2. The paper is a structural calibration model, not a reduced-form empirical paper

Even with the raw IHDS data in hand, the body of the paper is a structural
model of land-market frictions in Indian agriculture:

- Estimate a farm production function and permanent-component farm TFP
  from IHDS panel data (`tfp.do`, `tfp_crops.do`, `tfp_gammas.do`).
- Calibrate a Restuccia-Rogerson-style misallocation model by state
  (`calibration.do`, `calibrated_economy.do`), drawing idiosyncratic shocks
  and solving for optimal rental decisions for each farmer.
- Run four counterfactual economies (`counterfactuals.do`,
  `counterfactuals_gammas.do`): calibrated baseline, efficient-allocation
  without frictions, no state-level frictions, no idiosyncratic frictions.

Tables 1–5 and Figures 1–3, 6–11 are all outputs of this calibration /
counterfactual pipeline, not regression coefficients that can be sanity-
checked with a single side-by-side table. Even where the paper does report
reduced-form stylized facts (e.g., rental participation by land size,
institutional context from the Agriculture Census), those numbers also
depend on the cleaned IHDS panel and cannot be produced without the raw
microdata.

Replicating the full calibration in Python within the scope of this driver
(one-paper-at-a-time, no interactive ICPSR login, no multi-hour
optimization runs) is not realistic. The package is complete and the code
is well-structured — a researcher with IHDS credentials and Stata/SE 14.2
could run `master.do` end-to-end in roughly 1h15m on a workstation per the
README, but that is outside what this driver can do autonomously.

### 3. What a motivated replicator would do

- Register at ICPSR, accept IHDS terms of use, download the specified DS
  files for waves I and II, and the household-linking text file.
- Install Stata/SE 14.2 (the code is authored against that version; minor
  syntax differences may arise in newer releases).
- Run `master.do` from the package root after editing the `cd` line at the
  top.
- Check the produced `Tables/` and `Figures/` against the published paper.
  The README does not flag any known discrepancies, and the code is
  organized cleanly by phase (Clean → Estimation → Counterfactuals →
  TablesFigures), which suggests reproduction should be straightforward
  given the data.

No bug was found — the code was not run.
