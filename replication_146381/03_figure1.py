"""Figure 1: treated vs synthetic control trends under DID, SC, SDID."""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    load_california, panel_matrices, HERE,
    synthdid_estimate, sc_estimate, did_estimate,
)

OUT = HERE / "outputs"


def main():
    df = load_california()
    setup = panel_matrices(df)
    Y, N0, T0 = setup["Y"], setup["N0"], setup["T0"]
    times = setup["times"]

    ests = {
        "DID":  did_estimate(Y, N0, T0),
        "SC":   sc_estimate(Y, N0, T0),
        "SDID": synthdid_estimate(Y, N0, T0),
    }
    treated_traj = Y[N0:].mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (name, est) in zip(axes, ests.items()):
        omega = est["omega"]
        lam = est["lam"]
        synth = omega @ Y[:N0]
        # Intercept adjustment so pre-period synth matches treated (over lambda-weighted pre period)
        if name == "SDID":
            pre_diff = (treated_traj[:T0] * lam).sum() - (synth[:T0] * lam).sum()
        else:
            pre_diff = treated_traj[:T0].mean() - synth[:T0].mean() if name == "DID" else 0.0
        synth_shift = synth + pre_diff
        ax.plot(times, treated_traj, color="#27aeef", lw=1.5, label="California")
        ax.plot(times, synth_shift, color="#e35d5d", lw=1.5, label="Synthetic control")
        ax.axvline(times[T0] - 0.5, color="black", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("Year")
        if name == "DID":
            ax.set_ylabel("Packs per capita")
        # annotate ATT
        ax.text(0.02, 0.05, f"τ = {est['tau']:+.2f}",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    axes[-1].legend(loc="upper right", fontsize=9)
    plt.suptitle("Figure 1 replication: California Prop 99 — "
                 "treated vs synthetic control trends", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "figure1.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}/figure1.png")

    # Save lambda (time) weights over pre-period
    with open(OUT / "lambda_sdid.json", "w") as f:
        json.dump({"year": list(map(int, times[:T0])),
                   "lambda": ests["SDID"]["lam"].tolist()}, f, indent=2)

    # Print key lambda weights (top years)
    lam = ests["SDID"]["lam"]
    order = np.argsort(-lam)
    print("\nTop pre-period years by SDID time weight:")
    for i in order[:5]:
        if lam[i] > 0:
            print(f"  {times[i]}: λ = {lam[i]:.3f}")

    omega = ests["SDID"]["omega"]
    u_order = np.argsort(-omega)
    print("\nTop control states by SDID unit weight:")
    for i in u_order[:10]:
        if omega[i] > 0:
            print(f"  {setup['units'][i]}: ω = {omega[i]:.3f}")
    print(f"Non-zero controls: {int((omega > 1e-10).sum())}/{N0}")


if __name__ == "__main__":
    main()
