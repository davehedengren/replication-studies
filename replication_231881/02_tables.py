"""Phase 2: Tables 2 (summary stats) and 3 (DiD)."""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from utils import OUT, PUBLISHER_NAMES


def fmt(x, d=3):
    return "     NA" if pd.isna(x) else f"{x:7.{d}f}"


def table2(wk):
    """Summary statistics by store/publisher/agency status."""
    print("\n" + "=" * 100)
    print("Table 2: Summary Statistics (replication)")
    print("=" * 100)
    # tt: tag one obs per ebookid
    tt = wk.drop_duplicates("ebookid")
    order = [1, 2, 3, 4, 5, 0]

    def row_prices(store, agency_val):
        row = []
        for pub in order:
            mask = (wk["store"] == store) & (wk["soldby"] == pub) & (wk["agency"] == agency_val)
            if mask.sum() == 0:
                row.append((np.nan, np.nan))
            else:
                row.append((wk.loc[mask, "price"].mean(), wk.loc[mask, "price"].std()))
        return row

    header = ["HarperC", "Hachette", "S&S", "Macmillan", "PRHouse", "Other"]
    print("\nPrice e-book (wholesale)")
    for s, name in [(1, "Amazon"), (2, "B&N")]:
        r = row_prices(s, 0)
        line = f"  {name:<6}  " + "  ".join(
            f"{m:6.2f} ({sd:4.2f})" if pd.notna(m) else "      --     " for m, sd in r
        )
        print(line)

    print("\nPrice e-book (agency)")
    for s, name in [(1, "Amazon"), (2, "B&N")]:
        r = row_prices(s, 1)[:5]  # no 'other' in agency
        line = f"  {name:<6}  " + "  ".join(
            f"{m:6.2f} ({sd:4.2f})" if pd.notna(m) else "      --     " for m, sd in r
        ) + "       --"
        print(line)

    print("\nRating (by publisher, store-pooled)")
    r = []
    for pub in order:
        m = wk.loc[wk["soldby"] == pub, "rating"]
        r.append((m.mean(), m.std()))
    line = "           " + "  ".join(f"{m:6.2f} ({sd:4.2f})" for m, sd in r)
    print(line)

    print("\nTitles per publisher (ebookid-level)")
    line = "           " + "  ".join(
        f"{(tt['soldby'] == p).sum():6d}      " for p in order
    )
    print(line)
    print("\nObservations per publisher")
    line = "           " + "  ".join(
        f"{(wk['soldby'] == p).sum():7d}     " for p in order
    )
    print(line)

    # Save to csv
    rows = []
    for s, sname in [(1, "Amazon"), (2, "B&N")]:
        for pub in order:
            for ag in (0, 1):
                if pub == 0 and ag == 1:
                    continue
                m = (wk["store"] == s) & (wk["soldby"] == pub) & (wk["agency"] == ag)
                if m.sum() == 0:
                    continue
                rows.append({
                    "store": sname,
                    "publisher": PUBLISHER_NAMES[pub],
                    "agency": ag,
                    "mean_price": wk.loc[m, "price"].mean(),
                    "sd_price": wk.loc[m, "price"].std(),
                    "n": int(m.sum()),
                })
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT / "table2.csv", index=False)
    print(f"\nwrote {OUT/'table2.csv'}")


def _did_spec(panel, add_rating=False):
    """Run the DiD spec with ebookid + wkt FE, cluster SE on ebookid.

    panel must be filtered to a single store already.
    """
    df = panel.copy()
    # interaction dummies agency × soldby for soldby ∈ 1..5
    for pub in range(1, 6):
        df[f"ag_p{pub}"] = ((df["agency"] == 1) & (df["soldby"] == pub)).astype(float)

    df = df.set_index(["ebookid", "wkt"])  # (entity, time)
    regressors = [f"ag_p{p}" for p in range(1, 6)]
    if add_rating:
        regressors.append("rating")
    X = df[regressors]
    y = df["lprice"]
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return res


def _agg_effect(res, panel):
    """Aggregate effect: mean of the 5 publisher DiDs weighted by N in agency=1 subpop.

    Mirrors Stata's `margins r.agency, subpop(if agency==1)`.
    """
    # Weights = share of treated (agency==1) obs falling in each soldby group
    treated = panel[panel["agency"] == 1]
    weights = np.array([
        (treated["soldby"] == p).sum() / len(treated) for p in range(1, 6)
    ])
    coefs = np.array([res.params[f"ag_p{p}"] for p in range(1, 6)])
    agg = float(weights @ coefs)
    # Delta-method SE: w' V w restricted to the 5 DiD coefficients
    V = res.cov.loc[
        [f"ag_p{p}" for p in range(1, 6)],
        [f"ag_p{p}" for p in range(1, 6)],
    ].values
    agg_se = float(np.sqrt(weights @ V @ weights))
    return agg, agg_se


def table3(wk):
    print("\n" + "=" * 100)
    print("Table 3: Difference-in-differences analysis (replication)")
    print("=" * 100)
    rows = []
    results = {}
    for store in (1, 2):
        sub = wk[wk["store"] == store]
        r_nocov = _did_spec(sub, add_rating=False)
        r_cov = _did_spec(sub, add_rating=True)
        agg_nocov = _agg_effect(r_nocov, sub)
        agg_cov = _agg_effect(r_cov, sub)
        results[(store, "A")] = r_nocov
        results[(store, "B")] = r_cov
        for pub in range(1, 6):
            row = {
                "store": "Amazon" if store == 1 else "B&N",
                "publisher": PUBLISHER_NAMES[pub],
                "did_A": r_nocov.params[f"ag_p{pub}"],
                "se_A": r_nocov.std_errors[f"ag_p{pub}"],
                "did_B": r_cov.params[f"ag_p{pub}"],
                "se_B": r_cov.std_errors[f"ag_p{pub}"],
            }
            rows.append(row)
        rows.append({
            "store": "Amazon" if store == 1 else "B&N",
            "publisher": "__aggregate__",
            "did_A": agg_nocov[0], "se_A": agg_nocov[1],
            "did_B": agg_cov[0], "se_B": agg_cov[1],
        })
        rows.append({
            "store": "Amazon" if store == 1 else "B&N",
            "publisher": "__N__",
            "did_A": r_nocov.nobs, "se_A": np.nan,
            "did_B": r_cov.nobs, "se_B": np.nan,
        })
        rows.append({
            "store": "Amazon" if store == 1 else "B&N",
            "publisher": "__R2__",
            "did_A": r_nocov.rsquared, "se_A": np.nan,
            "did_B": r_cov.rsquared, "se_B": np.nan,
        })

    t3 = pd.DataFrame(rows)
    t3.to_csv(OUT / "table3.csv", index=False)
    print(t3.to_string(index=False))

    # Pretty print analogous to the Stata table
    print("\n" + "-" * 100)
    print(f"{'':35}{'Amazon':>30}{'Barnes & Noble':>33}")
    print(f"{'':35}{'(A)':>15}{'(B)':>15}{'(A)':>15}{'(B)':>15}")
    print("-" * 100)
    for pub in range(1, 6):
        name = f"Agency × {PUBLISHER_NAMES[pub]}"
        cells = []
        for store, spec in [(1, "A"), (1, "B"), (2, "A"), (2, "B")]:
            res = results[(store, spec)]
            b = res.params[f"ag_p{pub}"]
            se = res.std_errors[f"ag_p{pub}"]
            cells.append(f"{b:6.3f} ({se:.3f})")
        print(f"{name:<35}" + "".join(f"{c:>15}" for c in cells))

    # Aggregate + N
    for label, get in [
        ("Aggregate", lambda st, sp: _agg_effect(results[(st, sp)], wk[wk["store"] == st])),
    ]:
        cells = []
        for store, spec in [(1, "A"), (1, "B"), (2, "A"), (2, "B")]:
            ae, se = get(store, spec)
            cells.append(f"{ae:6.3f} ({se:.3f})")
        print(f"{'Aggregate agency effect':<35}" + "".join(f"{c:>15}" for c in cells))
    cells = []
    for store, spec in [(1, "A"), (1, "B"), (2, "A"), (2, "B")]:
        res = results[(store, spec)]
        cells.append(f"{int(res.nobs):,}")
    print(f"{'N':<35}" + "".join(f"{c:>15}" for c in cells))
    cells = []
    for store, spec in [(1, "A"), (1, "B"), (2, "A"), (2, "B")]:
        res = results[(store, spec)]
        cells.append(f"{res.rsquared:.3f}")
    print(f"{'R-squared (within)':<35}" + "".join(f"{c:>15}" for c in cells))
    print("-" * 100)

    # Key text numbers
    agg_amz = _agg_effect(results[(1, "A")], wk[wk["store"] == 1])
    agg_bn = _agg_effect(results[(2, "A")], wk[wk["store"] == 2])
    print(f"\nAmazon aggregate Δlog(price) = {agg_amz[0]:.3f} → pct change ≈ {(np.exp(agg_amz[0])-1)*100:.1f}%")
    print(f"B&N    aggregate Δlog(price) = {agg_bn[0]:.3f} → pct change ≈ {(np.exp(agg_bn[0])-1)*100:.1f}%")
    return results


def main():
    wk = pd.read_parquet(OUT / "ebook2_2.parquet")
    print(f"panel: {wk.shape[0]:,} rows")
    table2(wk)
    table3(wk)


if __name__ == "__main__":
    main()
