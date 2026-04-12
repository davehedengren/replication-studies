"""Load and clean the merged sales panel; persist an intermediate parquet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import build_merged_panel, load_merged_sales

OUT = Path(__file__).resolve().parent / "panel.parquet"


def main():
    raw = load_merged_sales()
    print(f"raw merged: {raw.shape}")
    print(f"unique villages: {raw['proper_village'].nunique()}")
    print(f"date range: {raw['date'].min()} to {raw['date'].max()}")

    panel = build_merged_panel(raw)
    print(f"\npanel (hh-month): {panel.shape}")
    print(f"unique villages: {panel['village'].nunique()}")
    print(f"unique households: {panel['only_digits_scard_id'].nunique()}")
    print("treatment distribution:")
    print(panel["treatment"].value_counts())
    print(f"\nControl mean bought: {panel.loc[panel['Control']==1, 'bought'].mean():.4f}")
    print(f"Control mean tot_consumption: {panel.loc[panel['Control']==1, 'tot_consumption'].mean():.4f}")

    panel.to_parquet(OUT)
    print(f"\nsaved → {OUT}")


if __name__ == "__main__":
    main()
