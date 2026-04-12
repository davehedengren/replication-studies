"""Table 1 / A1 in this paper are qualitative summaries of the literature
stored as Excel worksheets. They contain no regression or statistical output
to 'reproduce'. This script documents each sheet's structure and writes a
plain-text summary so the replication record is complete.
"""
import pandas as pd
from utils import TABLE_XLSX, OUT


def main():
    xl = pd.ExcelFile(TABLE_XLSX)
    lines = []
    lines.append(f"Workbook: {TABLE_XLSX.name}")
    lines.append(f"Sheets ({len(xl.sheet_names)}): {xl.sheet_names}")
    lines.append("")
    for s in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=s)
        lines.append(f"--- Sheet: {s} ---")
        lines.append(f"  shape: {df.shape}")
        lines.append(f"  columns: {list(df.columns)}")
        lines.append("")

    txt = "\n".join(lines)
    (OUT / "table1_a1_structure.txt").write_text(txt)
    print(txt)
    print("wrote:", OUT / "table1_a1_structure.txt")


if __name__ == "__main__":
    main()
