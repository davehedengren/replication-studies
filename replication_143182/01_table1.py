"""Replicate Table 1: Effect of quota exposure on decadal population change rates,
by sub-sample (urban / mining / rural / full) and outcome.

Reference: Table 1, Abramitzky et al. (2023), AEJ: Applied.
"""
from utils import (
    load_panel,
    build_did_panel,
    xtreg_fe,
    TABLE1_OUTCOMES,
    TABLE1_LABELS,
    SAMPLES,
)

PUBLISHED = {
    # (sample, outcome) -> (coef, se)
    ("urban", "qrtot"):  (-1.541, 0.610),
    ("urban", "nqrtot"): ( 2.595, 1.276),
    ("urban", "wtot"):   ( 1.903, 0.993),
    ("urban", "nwtot"):  ( 0.119, 0.185),
    ("urban", "fbold"):  (-0.0920, 0.220),
    ("urban", "fbnqr"):  ( 0.665, 0.193),
    ("mining","qrtot"):  (-1.278, 0.707),
    ("mining","nqrtot"): ( 1.177, 2.017),
    ("mining","wtot"):   ( 0.715, 1.516),
    ("mining","nwtot"):  ( 0.239, 0.179),
    ("mining","fbold"):  (-0.493, 0.339),
    ("mining","fbnqr"):  ( 0.717, 0.480),
    ("rural", "qrtot"):  (-0.720, 0.352),
    ("rural", "nqrtot"): (-0.781, 2.195),
    ("rural", "wtot"):   (-0.388, 1.812),
    ("rural", "nwtot"):  (-0.225, 0.246),
    ("rural", "fbold"):  (-0.403, 0.354),
    ("rural", "fbnqr"):  ( 0.236, 0.266),
    ("full",  "qrtot"):  (-1.578, 0.389),
    ("full",  "nqrtot"): ( 1.432, 0.924),
    ("full",  "wtot"):   ( 1.168, 0.693),
    ("full",  "nwtot"):  ( 0.0708,0.0861),
    ("full",  "fbold"):  (-0.345, 0.196),
    ("full",  "fbnqr"):  ( 0.539, 0.156),
}

PUBLISHED_N = {"urban": 340, "mining": 230, "rural": 348, "full": 918}


def run():
    df = load_panel(drop_outlier=True)
    panel = build_did_panel(df, TABLE1_OUTCOMES)

    rows = []
    for sample in ["urban", "mining", "rural", "full"]:
        if sample == "full":
            sub = panel
        else:
            sub = panel[panel["sea_type"] == SAMPLES[sample]]
        for y in TABLE1_OUTCOMES:
            est = xtreg_fe(sub, f"d{y}")
            pub_c, pub_se = PUBLISHED[(sample, y)]
            rows.append(
                {
                    "sample": sample,
                    "outcome": y,
                    "label": TABLE1_LABELS[y],
                    "coef": est["coef"],
                    "se": est["se"],
                    "n": est["n"],
                    "pub_coef": pub_c,
                    "pub_se": pub_se,
                    "pub_n": PUBLISHED_N[sample],
                }
            )
    return rows


def print_rows(rows):
    header = (
        f"{'Sample':<8} {'Outcome':<28} {'Repl coef':>11} {'(SE)':>10} "
        f"{'Pub coef':>11} {'(SE)':>10} {'N repl':>6} {'N pub':>6}  match"
    )
    print(header)
    print("-" * len(header))
    max_abs_diff = 0.0
    for r in rows:
        diff = r["coef"] - r["pub_coef"]
        max_abs_diff = max(max_abs_diff, abs(diff))
        mark = "OK" if abs(diff) < 5e-3 and abs(r["se"] - r["pub_se"]) < 5e-3 else "CHECK"
        print(
            f"{r['sample']:<8} {r['label']:<28} "
            f"{r['coef']:>11.4f} ({r['se']:>7.4f}) "
            f"{r['pub_coef']:>11.4f} ({r['pub_se']:>7.4f}) "
            f"{r['n']:>6d} {r['pub_n']:>6d}  {mark}"
        )
    print(f"\nMax absolute coef difference vs published: {max_abs_diff:.4f}")


if __name__ == "__main__":
    rows = run()
    print_rows(rows)
