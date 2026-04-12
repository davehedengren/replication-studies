"""Table B.II: Deviations from the Inflation Target (OLS with country FE, and
country+year FE; five columns).
"""
import os
import pandas as pd
from utils import OUT, fit_ols_fe

PUBLISHED = {
    # (coef, se, signif)  — "*" significance as reported in paper
    "I":   {"Revenue": (0.171, 0.076),  "Gross.Debt": (0.069, 0.035),
            "DebtRev": (-0.194, 0.099), "Center.Target": (-0.403, 0.063)},
    "II":  {"Revenue": (0.098, 0.076),  "Gross.Debt": (0.074, 0.034),
            "DebtRev": (-0.168, 0.096), "Center.Target": (-0.458, 0.062)},
    "III": {"Revenue": (0.063, 0.078),  "Gross.Debt": (0.073, 0.034),
            "DebtRev": (-0.149, 0.096), "Center.Target": (-0.441, 0.062),
            "GDP.Gap": (0.363, 0.102)},
    "IV":  {"Revenue": (0.125, 0.070),  "Gross.Debt": (0.062, 0.031),
            "DebtRev": (-0.163, 0.088), "Center.Target": (-0.360, 0.059),
            "Reer.YoY": (-13.956, 1.648)},
    "V":   {"Revenue": (0.087, 0.072),  "Gross.Debt": (0.058, 0.032),
            "DebtRev": (-0.136, 0.089), "Center.Target": (-0.342, 0.058),
            "GDP.Gap": (0.342, 0.095),  "Reer.YoY": (-13.645, 1.653)},
}

PUB_R2 = {"I": 0.290, "II": 0.408, "III": 0.433, "IV": 0.515, "V": 0.537}
PUB_N = {"I": 382, "II": 382, "III": 374, "IV": 372, "V": 364}

SPECS = [
    ("I",   None,                        False),
    ("II",  None,                        True),
    ("III", ["GDP.Gap"],                 True),
    ("IV",  ["Reer.YoY"],                True),
    ("V",   ["GDP.Gap", "Reer.YoY"],     True),
]

def main():
    df = pd.read_csv(os.path.join(OUT, "panel.csv"))

    rows = []
    for label, extras, year_fe in SPECS:
        _, coef, r2, n = fit_ols_fe(df, "deviation", extras=extras, year_fe=year_fe)
        pub = PUBLISHED[label]
        row = {"spec": label, "N": n, "R2": round(r2, 3), "pub_N": PUB_N[label],
               "pub_R2": PUB_R2[label]}
        for k in ("Revenue", "Gross.Debt", "DebtRev", "Center.Target",
                  "GDP.Gap", "Reer.YoY"):
            b, se = coef.get(k, (None, None))
            row[f"{k}_b"] = b if b is None else round(b, 3)
            row[f"{k}_se"] = se if se is None else round(se, 3)
            if k in pub:
                pb, pse = pub[k]
                row[f"{k}_pub_b"] = pb
                row[f"{k}_pub_se"] = pse
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "table_b2.csv"), index=False)

    print("Table B.II: Deviations from the Inflation Target")
    print("=" * 90)
    header = f"{'var':15s}  " + "  ".join(f"{s:>18s}" for s, _, _ in SPECS)
    print(header)
    for var in ("Revenue", "Gross.Debt", "DebtRev", "Center.Target", "GDP.Gap", "Reer.YoY"):
        line = f"{var:15s}  "
        for label, _, _ in SPECS:
            row = out[out["spec"] == label].iloc[0]
            b = row[f"{var}_b"]
            se = row[f"{var}_se"]
            pub = PUBLISHED[label].get(var)
            if b is None or pd.isna(b):
                cell = "—".center(18)
            else:
                tag = ""
                if pub is not None:
                    pb, _ = pub
                    if abs(b - pb) < 0.0015:
                        tag = "✓"
                    else:
                        tag = f"Δ{b-pb:+.3f}"
                cell = f"{b:+.3f}({se:.3f}) {tag}"
            line += f"{cell:>18s}  "
        print(line)

    print()
    print(f"{'N':15s}  " + "  ".join(
        f"{int(out[out['spec']==l].iloc[0]['N']):>6d}/{PUB_N[l]:<6d}      " for l, _, _ in SPECS))
    print(f"{'R2':15s}  " + "  ".join(
        f"{out[out['spec']==l].iloc[0]['R2']:>6.3f}/{PUB_R2[l]:<6.3f}      " for l, _, _ in SPECS))


if __name__ == "__main__":
    main()
