"""Shared helpers for replication 216641 (Araujo et al., AEJ:Macro 2024)."""
import os
import numpy as np
import pandas as pd

PKG = "/Users/davehedengren/code/replication-studies/216641-V1/ReplicationPackage/appendix/empirics"
OUT = "/Users/davehedengren/code/replication-studies/replication_216641"

DATABASE_XLSX = os.path.join(PKG, "database_paper.xlsx")
TARGET_XLSX = os.path.join(PKG, "target_data.xlsx")
GDP_XLSX = os.path.join(PKG, "gdp_quarterly_data.xlsx")

RHS_BASE = ["Revenue", "Gross.Debt", "DebtRev", "Center.Target"]


def load_database():
    df = pd.read_excel(DATABASE_XLSX, sheet_name="data")
    df.columns = [c.replace(" ", ".") for c in df.columns]
    df["deviation"] = df["CPI"] - df["Center.Target"]
    df["DebtRev"] = df["Gross.Debt"] * df["Revenue"] / 100.0
    return df


def build_overshoot():
    """Reproduce aux_overshoot_variable.R. Returns long df [Country, year, Overshoot_y]."""
    t = pd.read_excel(TARGET_XLSX, sheet_name="target", index_col=0)
    ub = pd.read_excel(TARGET_XLSX, sheet_name="upper_bound", index_col=0)
    cpi = pd.read_excel(TARGET_XLSX, sheet_name="cpi", index_col=0)
    overshoot_num = cpi - ub
    avg_upper_diff = (ub - t).stack().mean()
    # UK and Norway have point targets; paper uses center + 1.2 (rounded from ~1.17)
    for c in ("United.Kingdom", "Norway"):
        overshoot_num.loc[c] = cpi.loc[c] - (t.loc[c] + 1.2)
    # NB: `(df > 0).astype(float)` silently coerces NaN comparisons to False.
    # R's `overshoot_num > 0` preserves NA, so we reintroduce NaN explicitly.
    ov_bool = (overshoot_num > 0).astype(float)
    ov_bool = ov_bool.where(overshoot_num.notna())
    long = ov_bool.reset_index().melt(id_vars=ov_bool.index.name or "index",
                                      var_name="year", value_name="Overshoot_y")
    long = long.rename(columns={ov_bool.index.name or "index": "Country"})
    long["year"] = long["year"].astype(int)
    return long, avg_upper_diff


def build_gdp_gap_precomputed():
    """Use the precomputed quarterly GDP gap that ships in target_data.xlsx.

    This sheet is the output of running the R pipeline end-to-end (X-13 SA for
    Peru/Turkey + HP filter lambda=1600 applied to every country, aggregated
    quarterly). Using it directly gives exact numerical match to the published
    Table B.II column III/V regressions. We verified column III reproduces
    Revenue=0.063, Debt=0.073, DebtRev=-0.149, Target=-0.441, GDP.Gap=0.363
    with SEs and R² matching the paper to three decimals.
    """
    df = pd.read_excel(TARGET_XLSX, sheet_name="gdp gap quarterly")
    df = df.rename(columns={df.columns[0]: "year", df.columns[1]: "date"})
    annual = df.drop(columns=["date"]).groupby("year").mean()
    long = annual.stack().reset_index()
    long.columns = ["year", "Country", "GDP.Gap"]
    return long


def build_gdp_gap_stl():
    """STL-based reconstruction of the GDP gap.

    Used in the data audit (04_data_audit.py) to sanity-check against the
    precomputed ground truth. The X-13 ARIMA binary that the R code uses for
    seasonally adjusting Peru and Turkey is not in the shared venv, so we
    substitute statsmodels' additive seasonal_decompose (period=4) for those
    two countries and then apply the same HP filter (lambda=1600).
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.filters.hp_filter import hpfilter

    nsa = pd.read_excel(GDP_XLSX, sheet_name="NSA")
    sa = pd.read_excel(GDP_XLSX, sheet_name="SA")
    nsa = nsa.rename(columns={nsa.columns[0]: "Quarter"}).set_index("Quarter")
    sa = sa.rename(columns={sa.columns[0]: "Quarter"}).set_index("Quarter")

    for country in ("Peru", "Turkey"):
        series = nsa[country].dropna().astype(float)
        # Additive seasonal decomposition; SA = observed - seasonal.
        decomp = seasonal_decompose(series.values, period=4, model="additive",
                                    extrapolate_trend="freq")
        sa_vals = series.values - decomp.seasonal
        sa.loc[series.index, country] = sa_vals

    # HP filter on each country
    gap = pd.DataFrame(index=sa.index, columns=sa.columns, dtype=float)
    for col in sa.columns:
        s = sa[col].dropna().astype(float)
        if len(s) == 0:
            continue
        cycle, _ = hpfilter(s.values, lamb=1600)
        gap.loc[s.index, col] = cycle

    # Aggregate quarterly to annual means (labels like "1996Q1")
    gap = gap.reset_index()
    gap["year"] = gap["Quarter"].str[:4].astype(int)
    annual = gap.drop(columns=["Quarter"]).groupby("year").mean()
    long = annual.stack().reset_index()
    long.columns = ["year", "Country", "GDP.Gap"]
    return long


def assemble_panel():
    """Build the regression panel by merging the three data sources."""
    data = load_database()
    ov, _ = build_overshoot()
    gap = build_gdp_gap_precomputed()
    data = data.merge(ov, on=["Country", "year"], how="left")
    data = data.merge(gap, on=["Country", "year"], how="left")
    return data


def r_style_r2(y, resid):
    """Match R's `lm(... - 1)` R²: 1 - SSR/sum(y^2) (uncentered)."""
    y = np.asarray(y, dtype=float)
    ssr = float(np.sum(resid ** 2))
    sst = float(np.sum(y ** 2))
    return 1.0 - ssr / sst


def fit_ols_fe(df, outcome, extras=None, year_fe=False):
    """Fit OLS with country FE (and optional year FE), matching main.R formulas.

    Returns the fitted results plus a dict of (coef, se) for the rhs_base + extras.
    """
    import statsmodels.api as sm

    cols = ["Country", "year", outcome] + RHS_BASE + list(extras or [])
    d = df.dropna(subset=[c for c in cols if c != "Country" and c != "year"]).copy()
    X_parts = [d[RHS_BASE].reset_index(drop=True)]
    if extras:
        X_parts.append(d[list(extras)].reset_index(drop=True))
    country_dum = pd.get_dummies(d["Country"], drop_first=False, dtype=float).reset_index(drop=True)
    X_parts.append(country_dum)
    if year_fe:
        # Match R's `+ factor(year) - 1` which would double-count an intercept;
        # since we already have no intercept, drop first year dummy to avoid
        # collinearity with the country dummies sum.
        year_dum = pd.get_dummies(d["year"], drop_first=True, dtype=float).reset_index(drop=True)
        year_dum.columns = [f"year_{c}" for c in year_dum.columns]
        X_parts.append(year_dum)
    X = pd.concat(X_parts, axis=1)
    y = d[outcome].values
    model = sm.OLS(y, X.values).fit()
    names = list(X.columns)
    idx = {n: i for i, n in enumerate(names)}
    coef = {k: (model.params[idx[k]], model.bse[idx[k]]) for k in RHS_BASE + list(extras or [])}
    r2 = r_style_r2(y, model.resid)
    return model, coef, r2, int(model.nobs)


def fit_logit_fe(df, outcome, extras=None, year_fe=False):
    """Fit logit with country FE (and optional year FE), matching main.R glm() calls.

    Uses statsmodels GLM with Binomial family (IRLS) rather than `Logit.fit`
    because several countries in the sample have zero within-country variation
    in `Overshoot_y` (Norway, Sweden). `Logit.fit` fails to invert the Hessian
    in that case, while IRLS converges the main-regressor coefficients exactly
    (the separated country FE go to large negatives and contribute zero
    weight). This matches R's `glm(binomial("logit"))` behavior.
    """
    import statsmodels.api as sm

    cols = ["Country", "year", outcome] + RHS_BASE + list(extras or [])
    d = df.dropna(subset=[c for c in cols if c != "Country" and c != "year"]).copy()

    X_parts = [d[RHS_BASE].reset_index(drop=True)]
    if extras:
        X_parts.append(d[list(extras)].reset_index(drop=True))
    country_dum = pd.get_dummies(d["Country"], drop_first=False, dtype=float).reset_index(drop=True)
    X_parts.append(country_dum)
    if year_fe:
        year_dum = pd.get_dummies(d["year"], drop_first=True, dtype=float).reset_index(drop=True)
        year_dum.columns = [f"year_{c}" for c in year_dum.columns]
        X_parts.append(year_dum)
    X = pd.concat(X_parts, axis=1)
    y = d[outcome].astype(float).values

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y, X.values, family=sm.families.Binomial()).fit(maxiter=200)
    names = list(X.columns)
    idx = {n: i for i, n in enumerate(names)}
    coef = {k: (model.params[idx[k]], model.bse[idx[k]]) for k in RHS_BASE + list(extras or [])}
    return model, coef, float(model.llf), int(model.nobs)
