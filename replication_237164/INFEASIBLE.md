# Paper 237164: INFEASIBLE FOR REPLICATION

**Reason**: Requires R (not available on system) to compute higher-order risk preference measures via custom B-spline utility fitting code (HigherOrderRiskPreferences_v1.0.R). These HORP measures are the key independent variables in all regression tables. Without R, the analysis data cannot be constructed from the raw CSV.

**Additional challenges**: Complex PCA construction per grade for field behavior indices, bootstrap clustered SEs (1000 reps), Romano-Wolf multiple hypothesis testing correction.
