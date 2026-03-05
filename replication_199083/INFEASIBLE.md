# Paper 199083: INFEASIBLE FOR REPLICATION

**Reason**: The compiled analysis dataset (`data/process/final/3.compile.dta`) does not exist and must be built from raw data through 18 sequential build scripts. The data pipeline requires merging patent data, manufacturing census data, NASA contractor data, and numerous crosswalks. Additionally, the main regressions use `reghdfe` with 3-way fixed effects (county x industry x year) and 2-way clustering (MSA x industry), which adds significant implementation complexity.

**Available**: Raw data files (.dta) for patents, crosswalks, and some auxiliary sources.
**Missing**: All intermediate and final analysis datasets.
