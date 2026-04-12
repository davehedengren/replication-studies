"""Data audit for the CDL panel used in Cormun & De Leo."""
from __future__ import annotations

import numpy as np
import pandas as pd

import utils as u


def main() -> None:
    cdl = u.load_cdl()
    names = u.load_country_names()

    T_total = cdl.shape[0]
    print(f"Raw CDL panel: rows={T_total} (monthly, 1974m1..~2010)")

    rows = []
    for ord1 in u.N_MATLAB:
        block = u.country_block(cdl, ord1)
        clean = u.drop_nan_rows(block)
        any_nan = np.any(np.isnan(block), axis=1)
        first_ok = int(np.argmax(~any_nan)) if (~any_nan).any() else -1
        last_ok = int(T_total - 1 - np.argmax(~any_nan[::-1])) if (~any_nan).any() else -1
        ffr, hr, fx = clean[:, 0], clean[:, 1], clean[:, 2]
        rows.append(
            dict(
                ord1=ord1,
                country=names[ord1 - 1],
                n_obs=clean.shape[0],
                first=first_ok,
                last=last_ok,
                ffr_min=ffr.min(),
                ffr_max=ffr.max(),
                hr_min=hr.min(),
                hr_max=hr.max(),
                fx_min=fx.min(),
                fx_max=fx.max(),
                fx_std=fx.std(),
                hr_neg=int((hr < 0).sum()),
                ffr_neg=int((ffr < 0).sum()),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(u.OUT / "data_audit.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print()
    print("Panel balance:")
    print(f"  shortest sample: {df['n_obs'].min()} months ({df.loc[df['n_obs'].idxmin(), 'country']})")
    print(f"  longest  sample: {df['n_obs'].max()} months ({df.loc[df['n_obs'].idxmax(), 'country']})")
    print(f"  median sample  : {df['n_obs'].median():.0f} months")

    print()
    print("Interest-rate plausibility checks:")
    if (df['hr_neg'] > 0).any():
        print("  negative home rates in:",
              df.loc[df['hr_neg'] > 0, ['country', 'hr_neg']].to_string(index=False))
    else:
        print("  no negative home rates (sample ends before 2015 NIRP period)")
    if (df['ffr_neg'] > 0).any():
        print("  ⚠️ negative FFR values present")
    else:
        print("  no negative FFR values")

    # Check panel for duplicates in time index
    year = pd.to_numeric(cdl.iloc[:, 0], errors='coerce')
    month = pd.to_numeric(cdl.iloc[:, 1], errors='coerce')
    dates = pd.to_datetime(dict(year=year, month=month, day=1))
    dupes = dates.duplicated().sum()
    print(f"Time index duplicates: {dupes}")
    print(f"Time index range: {dates.min()} .. {dates.max()}")


if __name__ == "__main__":
    main()
