"""Build the merged panel used for Tables B.I–B.III and save it to CSV."""
import os
import pandas as pd
from utils import assemble_panel, OUT

def main():
    df = assemble_panel()
    out = os.path.join(OUT, "panel.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}: shape={df.shape}")
    print("Countries:", df["Country"].nunique())
    print("Year range:", int(df["year"].min()), "-", int(df["year"].max()))
    print("Non-missing counts:")
    for c in ["CPI", "Center.Target", "Gross.Debt", "Revenue", "Overshoot_y", "GDP.Gap", "Reer.YoY"]:
        print(f"  {c:16s}: {df[c].notna().sum()} / {len(df)}")

if __name__ == "__main__":
    main()
