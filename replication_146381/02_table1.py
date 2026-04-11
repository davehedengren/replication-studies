"""Replicate Table 1 of Arkhangelsky et al. (2021): estimates and placebo SEs
for SDID, SC, DID, and DIFP on the California Prop 99 application. MC is
not replicated (requires cross-validated nuclear-norm matrix completion;
see writeup)."""
from __future__ import annotations
import json
import numpy as np
from utils import (
    load_california, panel_matrices, HERE,
    synthdid_estimate, sc_estimate, did_estimate, difp_estimate,
    placebo_se,
)

OUT = HERE / "outputs"

PUBLISHED = {
    "SDID": (-15.6, 8.4),
    "SC":   (-19.6, 9.9),
    "DID":  (-27.3, 17.7),
    "DIFP": (-11.1, 9.5),
    # MC:   (-20.2, 11.5)  # not replicated
}


def main():
    df = load_california()
    setup = panel_matrices(df)
    Y, N0, T0 = setup["Y"], setup["N0"], setup["T0"]

    estimators = {
        "SDID": synthdid_estimate,
        "SC":   sc_estimate,
        "DID":  did_estimate,
        "DIFP": difp_estimate,
    }
    results = {}
    for name, fn in estimators.items():
        est = fn(Y, N0, T0)
        se, _ = placebo_se(est, replications=200, seed=12345)
        results[name] = {"estimate": est["tau"], "se": se, "est": est}
        print(f"{name}: tau = {est['tau']:+.4f}, SE = {se:.4f}")

    print("\nTable 1 comparison (published vs replication)")
    print(f"{'Estimator':<6} | {'Pub est':>8} {'Pub SE':>7} | {'Our est':>8} {'Our SE':>7} | {'Δ est':>7}")
    print("-" * 66)
    rows = []
    for name in ["SDID", "SC", "DID", "DIFP"]:
        pub_est, pub_se = PUBLISHED[name]
        our_est = results[name]["estimate"]
        our_se = results[name]["se"]
        delta = our_est - pub_est
        print(f"{name:<6} | {pub_est:>8.1f} {pub_se:>7.1f} | {our_est:>8.1f} {our_se:>7.1f} | {delta:>+7.2f}")
        rows.append({
            "estimator": name,
            "pub_estimate": pub_est, "pub_se": pub_se,
            "our_estimate": our_est, "our_se": our_se,
            "delta_estimate": delta,
        })

    with open(OUT / "table1.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Save weights for figure / writeup
    sdid_weights = {
        "omega": results["SDID"]["est"]["omega"].tolist(),
        "lam":   results["SDID"]["est"]["lam"].tolist(),
    }
    sc_weights = {
        "omega": results["SC"]["est"]["omega"].tolist(),
        "lam":   results["SC"]["est"]["lam"].tolist(),
    }
    with open(OUT / "weights.json", "w") as f:
        json.dump({"SDID": sdid_weights, "SC": sc_weights,
                   "units": setup["units"][:N0], "times": list(map(int, setup["times"][:T0]))},
                  f, indent=2)

    print(f"\nSaved: {OUT}/table1.json, {OUT}/weights.json")


if __name__ == "__main__":
    main()
