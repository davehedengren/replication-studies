"""Replicate Tables 2 (age group) and 3 (industry group)."""
from utils import (
    load_panel,
    build_did_panel,
    xtreg_fe,
    SAMPLES,
)

# Published Table 2: age-group DiD (QE2 x Post)
TABLE2_PUB = {
    # (sample, y) -> (coef, se)
    ("urban", "y_qrtot"):  (-1.301, 0.528),
    ("urban", "y_nqrtot"): ( 1.733, 0.901),
    ("urban", "o_qrtot"):  (-0.240, 0.0845),
    ("urban", "o_nqrtot"): ( 0.863, 0.400),
    ("mining","y_qrtot"):  (-1.142, 0.617),
    ("mining","y_nqrtot"): ( 0.768, 1.389),
    ("mining","o_qrtot"):  (-0.136, 0.0934),
    ("mining","o_nqrtot"): ( 0.409, 0.658),
    ("rural", "y_qrtot"):  (-0.663, 0.310),
    ("rural", "y_nqrtot"): (-0.867, 1.658),
    ("rural", "o_qrtot"):  (-0.0568,0.0477),
    ("rural", "o_nqrtot"): ( 0.0859,0.591),
    ("full",  "y_qrtot"):  (-1.380, 0.339),
    ("full",  "y_nqrtot"): ( 0.865, 0.637),
    ("full",  "o_qrtot"):  (-0.198, 0.0522),
    ("full",  "o_nqrtot"): ( 0.567, 0.307),
}

TABLE2_OUTCOMES = ["y_qrtot", "y_nqrtot", "o_qrtot", "o_nqrtot"]

# Published Table 3: industry-group DiD (QE2 x Post)
# Order: (sample, y) -> (coef, se)
TABLE3_PUB = {
    ("urban", "qrtot_manuf"):        (-0.795, 0.412),
    ("urban", "qrtot_mining"):       ( 0.00281,0.0125),
    ("urban", "qrtot_agriculture"):  (-0.0431,0.0994),
    ("urban", "qrtot_other_ind"):    (-0.391, 0.109),
    ("urban", "qrtot_no_ind"):       (-0.315, 0.128),
    ("urban", "nqrtot_manuf"):       ( 0.700, 0.332),
    ("urban", "nqrtot_mining"):      (-0.0117,0.0621),
    ("urban", "nqrtot_agriculture"): ( 0.481, 0.282),
    ("urban", "nqrtot_other_ind"):   ( 1.011, 0.632),
    ("urban", "nqrtot_no_ind"):      ( 0.415, 0.580),

    ("mining","qrtot_manuf"):        ( 0.192, 0.245),
    ("mining","qrtot_mining"):       (-1.436, 0.262),
    ("mining","qrtot_agriculture"):  (-0.176, 0.0987),
    ("mining","qrtot_other_ind"):    ( 0.0569,0.246),
    ("mining","qrtot_no_ind"):       ( 0.0850,0.177),
    ("mining","nqrtot_manuf"):       ( 0.0905,0.327),
    ("mining","nqrtot_mining"):      (-0.438, 0.525),
    ("mining","nqrtot_agriculture"): ( 0.323, 0.626),
    ("mining","nqrtot_other_ind"):   ( 0.585, 0.779),
    ("mining","nqrtot_no_ind"):      ( 0.616, 0.704),

    ("rural", "qrtot_manuf"):        ( 0.106, 0.120),
    ("rural", "qrtot_mining"):       ( 0.00334,0.0230),
    ("rural", "qrtot_agriculture"):  (-0.607, 0.219),
    ("rural", "qrtot_other_ind"):    (-0.0697,0.142),
    ("rural", "qrtot_no_ind"):       (-0.152, 0.0638),
    ("rural", "nqrtot_manuf"):       ( 0.319, 0.347),
    ("rural", "nqrtot_mining"):      (-0.0770,0.0856),
    ("rural", "nqrtot_agriculture"): (-0.0768,1.003),
    ("rural", "nqrtot_other_ind"):   (-0.328, 0.824),
    ("rural", "nqrtot_no_ind"):      (-0.619, 0.711),

    ("full",  "qrtot_manuf"):        (-0.364, 0.191),
    ("full",  "qrtot_mining"):       (-0.728, 0.185),
    ("full",  "qrtot_agriculture"):  (-0.168, 0.0737),
    ("full",  "qrtot_other_ind"):    (-0.172, 0.101),
    ("full",  "qrtot_no_ind"):       (-0.146, 0.105),
    ("full",  "nqrtot_manuf"):       ( 0.317, 0.200),
    ("full",  "nqrtot_mining"):      (-0.370, 0.160),
    ("full",  "nqrtot_agriculture"): ( 0.518, 0.319),
    ("full",  "nqrtot_other_ind"):   ( 0.512, 0.405),
    ("full",  "nqrtot_no_ind"):      ( 0.455, 0.336),
}

TABLE3_OUTCOMES = [
    "qrtot_manuf", "qrtot_mining", "qrtot_agriculture", "qrtot_other_ind", "qrtot_no_ind",
    "nqrtot_manuf", "nqrtot_mining", "nqrtot_agriculture", "nqrtot_other_ind", "nqrtot_no_ind",
]


def _run(outcomes, pub_dict, title):
    df = load_panel(drop_outlier=True)
    panel = build_did_panel(df, outcomes)

    print(f"\n=== {title} ===")
    header = f"{'sample':<8} {'outcome':<22} {'repl coef':>11} {'(SE)':>10} {'pub coef':>11} {'(SE)':>10} {'diff':>9}"
    print(header)
    print("-" * len(header))
    max_abs_diff = 0.0
    for sample in ["urban", "mining", "rural", "full"]:
        sub = panel if sample == "full" else panel[panel["sea_type"] == SAMPLES[sample]]
        for y in outcomes:
            est = xtreg_fe(sub, f"d{y}")
            pc, ps = pub_dict[(sample, y)]
            diff = est["coef"] - pc
            max_abs_diff = max(max_abs_diff, abs(diff))
            print(
                f"{sample:<8} {y:<22} "
                f"{est['coef']:>11.4f} ({est['se']:>7.4f}) "
                f"{pc:>11.4f} ({ps:>7.4f}) "
                f"{diff:>9.4f}"
            )
    print(f"Max |coef diff|: {max_abs_diff:.4f}")


if __name__ == "__main__":
    _run(TABLE2_OUTCOMES, TABLE2_PUB, "Table 2 (age groups)")
    _run(TABLE3_OUTCOMES, TABLE3_PUB, "Table 3 (industry groups)")
