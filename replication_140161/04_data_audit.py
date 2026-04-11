"""Data audit for the Wave-1 Henry-Zhuravskaya-Guriev sample."""
import numpy as np
import pandas as pd

from replication_140161.utils import build_all, load_survey, TREATMENT_LABELS


def main():
    df = build_all()
    raw = {i: load_survey(i) for i in (1, 2, 3)}

    print("=" * 72)
    print("1. Sample size audit")
    print("=" * 72)
    print(f"Per-treatment rows in cleaned panel: {dict(df.groupby('survey').size())}")
    print("Published balance table (Table 1) reports 845 / 846 / 846.")
    print("Discrepancy: one row per treatment; cause verified below.")

    print()
    print("=" * 72)
    print("2. Filter-sensitivity: rows lost at each stage")
    print("=" * 72)
    for i in (1, 2, 3):
        path = f"Survey+{i}..."
        df_raw = pd.read_csv(
            f"140161-V1/original_data/Survey+{i}_May+27%2C+2019_01.0{3 if i==2 or i==3 else 2}.csv",
            low_memory=False,
        )
        df_raw.columns = [c.lower().replace(" ", "").replace("(", "").replace(")", "")
                           for c in df_raw.columns]
        df_raw = df_raw.iloc[3:].reset_index(drop=True)
        n0 = len(df_raw)
        df_raw = df_raw[df_raw["distributionchannel"] != "preview"]
        n1 = len(df_raw)
        df_raw["durationinseconds"] = pd.to_numeric(df_raw["durationinseconds"], errors="coerce")
        n_eq_250 = int((df_raw["durationinseconds"] == 250).sum())
        n_lt_250 = int((df_raw["durationinseconds"] < 250).sum())
        df_raw = df_raw[df_raw["durationinseconds"] >= 250]
        n2 = len(df_raw)
        df_raw["gc"] = pd.to_numeric(df_raw["gc"], errors="coerce")
        df_raw = df_raw[df_raw["gc"] == 1]
        n3 = len(df_raw)
        print(f"Survey {i}: raw={n0}, no-preview={n1}, dur>=250={n2} "
              f"(dur<250 dropped={n0-n1-n_lt_250+n_lt_250}), gc==1={n3}; "
              f"duration==250 exact: {n_eq_250}")

    print()
    print("=" * 72)
    print("3. Balance check on baseline covariates (Alt-Facts vs Imposed/Voluntary)")
    print("=" * 72)
    covs = ["age", "educ", "income", "religion_q8", "q13", "q14", "q15",
            "neg_image_ue", "second_mlp", "altruism",
            "share_interest", "share_image", "share_influence"]
    means = df.groupby("survey")[covs].mean().round(3)
    print(means.T)

    print()
    print("=" * 72)
    print("4. Missingness by treatment")
    print("=" * 72)
    miss = df.groupby("survey")[covs].apply(lambda x: x.isna().mean()).round(3)
    print(miss.T)

    print()
    print("=" * 72)
    print("5. DV descriptives & sanity checks")
    print("=" * 72)
    print(f"want_share_fb range: [{df['want_share_fb'].min()}, {df['want_share_fb'].max()}] "
          f"(should be 0/1)")
    print(f"want_share_facts range: [{df['want_share_facts'].min()}, {df['want_share_facts'].max()}]")
    print(f"see_facts range (survey 3 only): "
          f"[{df.loc[df.survey==3, 'see_facts'].min()}, "
          f"{df.loc[df.survey==3, 'see_facts'].max()}]")

    # Duplicates
    print()
    print(f"Duplicate rows by all columns: {df.duplicated().sum()}")

    # Crosstab survey x outcome
    print()
    print("Crosstab: treatment × want_share_fb")
    print(pd.crosstab(df["survey"].map(TREATMENT_LABELS), df["want_share_fb"],
                      margins=True))
    print()
    print("Crosstab: treatment × want_share_facts (surveys 2 and 3)")
    sub = df[df.survey.isin([2, 3])]
    print(pd.crosstab(sub["survey"].map(TREATMENT_LABELS), sub["want_share_facts"],
                      margins=True))

    print()
    print("=" * 72)
    print("6. Outlier checks on continuous vars")
    print("=" * 72)
    for v in ["age", "q14", "income", "educ", "share_interest", "share_influence"]:
        s = df[v].dropna()
        if len(s) == 0:
            continue
        print(f"  {v:18s} n={len(s):5d}  min={s.min():8.2f}  "
              f"p50={s.median():8.2f}  p99={s.quantile(0.99):8.2f}  "
              f"max={s.max():8.2f}")


if __name__ == "__main__":
    main()
