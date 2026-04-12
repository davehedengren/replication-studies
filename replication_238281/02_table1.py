"""Replicate Table 1: ITT effects of clean water offers on water orders."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import twoway_fe_cluster

PANEL = Path(__file__).resolve().parent / "panel.parquet"


def run(panel, y_col, regs):
    """Run several specs; print a Table-1-like layout."""
    out = []
    for name, X_cols in regs.items():
        X = panel[X_cols].values
        res = twoway_fe_cluster(panel[y_col].values, X, panel["village"].values, panel["month_idx"].values)
        out.append((name, X_cols, res))
    return out


def fmt_block(name, X_cols, res):
    print(f"\n=== {name}: y = {name.split('|')[0].strip()} ===")
    for var, b, s in zip(X_cols, res["coef"], res["se"]):
        print(f"  {var:25s} {b:8.4f}  ({s:.4f})")
    print(f"  N = {res['n']:,}")


def main():
    panel = pd.read_parquet(PANEL)

    # Table 1 has 4 columns: any orders (pooled / sub) and litres (pooled / sub)
    pooled = ["Discount", "Exchangeable", "FreeRation", "OneTime"]
    sub = [
        "Discount0.1", "Discount0.5", "Discount0.9",
        "Exchangeable0.1", "Exchangeable0.5", "Exchangeable0.9", "Exchangeable1",
        "FreeRation", "OneTime",
    ]

    regs_any = {"Col(1) any | pooled": pooled, "Col(2) any | sub": sub}
    regs_lit = {"Col(3) lit | pooled": pooled, "Col(4) lit | sub": sub}

    print("##### Table 1 — Any orders #####")
    for name, X_cols, res in run(panel, "bought", regs_any):
        fmt_block(name, X_cols, res)

    print("\n##### Table 1 — Orders in litres #####")
    for name, X_cols, res in run(panel, "tot_consumption", regs_lit):
        fmt_block(name, X_cols, res)

    print("\n--- Control means (estimation sample) ---")
    cm = panel["Control"] == 1
    print(f"  any orders: {panel.loc[cm, 'bought'].mean():.4f}")
    print(f"  litres    : {panel.loc[cm, 'tot_consumption'].mean():.4f}")

    # Save a tidy comparison table
    rows = []
    paper = {
        "Discount":          (0.38, 0.02, 95.93, 7.68),
        "Exchangeable":      (0.90, 0.02, 290.79, 11.12),
        "FreeRation":        (0.89, 0.02, 269.93, 6.96),
        "OneTime":           (0.14, 0.01, 13.13, 1.98),
        "Discount0.1":       (0.09, 0.03, 19.99, 9.31),
        "Discount0.5":       (0.15, 0.04, 34.08, 10.84),
        "Discount0.9":       (0.88, 0.02, 232.12, 7.22),
        "Exchangeable0.1":   (0.89, 0.03, 285.86, 14.12),
        "Exchangeable0.5":   (0.87, 0.04, 282.69, 14.56),
        "Exchangeable0.9":   (0.93, 0.02, 297.99, 10.18),
        "Exchangeable1":     (0.93, 0.02, 297.16, 11.22),
    }
    for name, X_cols, res in run(panel, "bought", regs_any):
        col = "(1)" if "pooled" in name else "(2)"
        for var, b, s in zip(X_cols, res["coef"], res["se"]):
            if var in paper:
                rows.append({"col": col, "var": var, "b": b, "se": s,
                             "paper_b": paper[var][0], "paper_se": paper[var][1]})
    for name, X_cols, res in run(panel, "tot_consumption", regs_lit):
        col = "(3)" if "pooled" in name else "(4)"
        for var, b, s in zip(X_cols, res["coef"], res["se"]):
            if var in paper:
                rows.append({"col": col, "var": var, "b": b, "se": s,
                             "paper_b": paper[var][2], "paper_se": paper[var][3]})

    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parent / "table1_comparison.csv", index=False)
    print("\n--- comparison vs paper ---")
    with pd.option_context("display.max_rows", None, "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
