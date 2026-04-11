"""Load California Prop 99 data and build the wide panel matrix."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from utils import load_california, panel_matrices, HERE

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    df = load_california()
    print(f"Raw rows: {len(df)}")
    print(f"Unique states: {df['State'].nunique()}")
    print(f"Year range: {df['Year'].min()}–{df['Year'].max()}")
    print(f"Treated state(s): {sorted(df[df['treated'] == 1]['State'].unique())}")
    print(f"First treated year: {df[df['treated'] == 1]['Year'].min()}")

    setup = panel_matrices(df)
    Y = setup["Y"]; N0 = setup["N0"]; T0 = setup["T0"]
    print(f"\nPanel matrix shape: {Y.shape}")
    print(f"N0 (controls) = {N0}, N1 (treated) = {Y.shape[0] - N0}")
    print(f"T0 (pre-treatment periods) = {T0}, T1 (post) = {Y.shape[1] - T0}")
    print(f"Last pre-treatment year: {setup['times'][T0 - 1]}")
    print(f"First treated year: {setup['times'][T0]}")
    print(f"Treated unit: {setup['units'][-1]}")

    np.savez(OUT / "panel.npz",
             Y=Y, N0=N0, T0=T0,
             units=np.array(setup["units"]),
             times=np.array(setup["times"]))
    with open(OUT / "panel_meta.json", "w") as f:
        json.dump({
            "N0": int(N0), "T0": int(T0),
            "N": int(Y.shape[0]), "T": int(Y.shape[1]),
            "units": setup["units"], "times": list(map(int, setup["times"])),
        }, f, indent=2)
    print(f"\nSaved to {OUT}/panel.npz")


if __name__ == "__main__":
    main()
