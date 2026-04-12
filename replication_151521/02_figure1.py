"""Reproduce Figure 1: College Tuition and Fees ($2016), 1990–91 to 2016–17."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_figure1, OUT


def main():
    df = load_figure1()

    print("Figure 1 key values (paper vs replication):")
    print(f"  1990–91 Public Sticker:   paper $3,520 | repl ${df['public_sticker'].iloc[0]:,.0f}")
    print(f"  1990–91 Public Net:       paper $2,000 | repl ${df['public_net'].iloc[0]:,.0f}")
    print(f"  1990–91 Private Sticker:  paper $17,240 | repl ${df['private_sticker'].iloc[0]:,.0f}")
    print(f"  1990–91 Private Net:      paper $11,750 | repl ${df['private_net'].iloc[0]:,.0f}")
    print(f"  2016–17 Public Sticker:   paper $9,650 | repl ${df['public_sticker'].iloc[-1]:,.0f}")
    print(f"  2016–17 Public Net:       paper $3,770 | repl ${df['public_net'].iloc[-1]:,.0f}")
    print(f"  2016–17 Private Sticker:  paper $33,480 | repl ${df['private_sticker'].iloc[-1]:,.0f}")
    print(f"  2016–17 Private Net:      paper $14,190 | repl ${df['private_net'].iloc[-1]:,.0f}")

    # Growth rates
    print("\nGrowth 1990–91 → 2016–17:")
    for col, lbl in [("public_sticker", "Public Sticker"),
                     ("public_net", "Public Net"),
                     ("private_sticker", "Private Sticker"),
                     ("private_net", "Private Net")]:
        g = df[col].iloc[-1] / df[col].iloc[0] - 1
        print(f"  {lbl:18s}: {100 * g:+.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    ax.plot(x, df["private_sticker"], marker="^", color="#7CB342", label="Private Sticker")
    ax.plot(x, df["private_net"], marker="o", color="#1E88E5", label="Private Net")
    ax.plot(x, df["public_sticker"], marker="x", color="#7E57C2", label="Public Sticker")
    ax.plot(x, df["public_net"], marker="s", color="#E53935", label="Public Net")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([y for y in df["year"][::2]], rotation=45)
    ax.set_title("Figure 1: College Tuition and Fees ($2016)")
    ax.set_ylabel("USD (2016)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = OUT / "figure1_replication.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")

    df.to_csv(OUT / "figure1_data.csv", index=False)


if __name__ == "__main__":
    main()
