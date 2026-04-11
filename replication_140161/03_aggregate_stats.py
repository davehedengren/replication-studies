"""Verify the descriptive numbers cited in Section 3 of the paper.

The paper text reports:
  - 4231 participants exposed to alt-facts, of whom 495 said yes to sharing
    (11.7%).  [These are Wave 1 totals *plus* the Alt-Facts arm of Wave 2.]
  - 130 of those 495 proceeded to the second click, 30 to the third.
  - Wave 1 mean shares: 14.7%, 10.2%, 10.8% (Alt-Facts, Imposed FC, Voluntary FC).
  - 39% of Voluntary Fact-Check participants chose to view the fact-check.

We can only verify the Wave 1 numbers from our reconstructed data (we do not
rebuild Wave 2).  See `writeup_140161.md` for context.
"""
from replication_140161.utils import build_all


def main():
    df = build_all()

    total_yes = int(df["want_share_fb"].sum())
    print(f"Wave-1 total 'yes' to sharing alt-facts: {total_yes}  (of {len(df)})")
    print(f"  share: {total_yes/len(df):.4f}")
    print()

    print("By-treatment shares (paper text p.13 / Table 2 col 1 constant):")
    print(df.groupby("survey")["want_share_fb"].mean().round(4))
    print()

    # Voluntary Fact-Check: fraction choosing to view fact-checking
    v = df[df["survey"] == 3]
    share_view = v["see_facts"].mean()
    print(f"Voluntary FC: share who chose to view fact-check: {share_view:.3f}  (paper: 0.39)")

    # Sharing of fact-check among imposed vs voluntary
    print("\nShare who said yes to sharing the fact-check (Table 3):")
    sub = df[df["survey"].isin([2, 3])]
    print(sub.groupby("survey")["want_share_facts"].mean().round(4))


if __name__ == "__main__":
    main()
