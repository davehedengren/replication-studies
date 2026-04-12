"""Table B.III: Probability of Overshooting the Target (logit with country FE,
and country+year FE; five columns).
"""
import os
import pandas as pd
from utils import OUT, fit_logit_fe

PUBLISHED = {
    "I":   {"Revenue": (0.145, 0.091),  "Gross.Debt": (0.034, 0.044),
            "DebtRev": (-0.114, 0.125), "Center.Target": (-0.624, 0.263)},
    "II":  {"Revenue": (0.108, 0.105),  "Gross.Debt": (0.055, 0.052),
            "DebtRev": (-0.107, 0.149), "Center.Target": (-1.242, 0.376)},
    "III": {"Revenue": (0.084, 0.109),  "Gross.Debt": (0.053, 0.053),
            "DebtRev": (-0.088, 0.151), "Center.Target": (-1.207, 0.376),
            "GDP.Gap": (0.206, 0.158)},
    "IV":  {"Revenue": (0.115, 0.110),  "Gross.Debt": (0.050, 0.053),
            "DebtRev": (-0.121, 0.151), "Center.Target": (-0.990, 0.390),
            "Reer.YoY": (-10.493, 3.062)},
    "V":   {"Revenue": (0.082, 0.113),  "Gross.Debt": (0.042, 0.054),
            "DebtRev": (-0.085, 0.154), "Center.Target": (-0.936, 0.386),
            "GDP.Gap": (0.218, 0.167),  "Reer.YoY": (-10.262, 3.101)},
}

PUB_N = {"I": 377, "II": 377, "III": 369, "IV": 368, "V": 360}
PUB_LL = {"I": -178.526, "II": -151.367, "III": -149.281, "IV": -139.619, "V": -137.954}

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
        _, coef, ll, n = fit_logit_fe(df, "Overshoot_y", extras=extras, year_fe=year_fe)
        pub = PUBLISHED[label]
        row = {"spec": label, "N": n, "LL": round(ll, 3),
               "pub_N": PUB_N[label], "pub_LL": PUB_LL[label]}
        for k in ("Revenue", "Gross.Debt", "DebtRev", "Center.Target", "GDP.Gap", "Reer.YoY"):
            b, se = coef.get(k, (None, None))
            row[f"{k}_b"] = b if b is None else round(b, 3)
            row[f"{k}_se"] = se if se is None else round(se, 3)
            if k in pub:
                pb, pse = pub[k]
                row[f"{k}_pub_b"] = pb
                row[f"{k}_pub_se"] = pse
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "table_b3.csv"), index=False)

    print("Table B.III: Probability of Overshooting the Target")
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
                    tol = 0.01 if abs(pb) > 1 else 0.0025
                    if abs(b - pb) < tol:
                        tag = "✓"
                    else:
                        tag = f"Δ{b-pb:+.3f}"
                cell = f"{b:+.3f}({se:.3f}) {tag}"
            line += f"{cell:>18s}  "
        print(line)

    print()
    print(f"{'N':15s}  " + "  ".join(
        f"{int(out[out['spec']==l].iloc[0]['N']):>6d}/{PUB_N[l]:<6d}      " for l, _, _ in SPECS))
    print(f"{'LogLik':15s}  " + "  ".join(
        f"{out[out['spec']==l].iloc[0]['LL']:>8.2f}/{PUB_LL[l]:<8.2f}  " for l, _, _ in SPECS))


if __name__ == "__main__":
    main()
