"""Load and validate all datasets used in the JEP paper."""
from utils import load, prep_pooled_age18to24, prep_pooled_cross_country

if __name__ == "__main__":
    names = [
        "pooled_age18to24",
        "pooled_cross_country",
        "pooled_admin1",
        "pooled_admin_young",
        "pooled_adminXwealth",
        "pooled_cohortJEP",
        "gender_age18to24",
        "ITU",
    ]
    for n in names:
        df = load(n)
        print(f"{n:28s} rows={df.shape[0]:>6d}  cols={df.shape[1]}")

    print()
    p = prep_pooled_age18to24()
    print("pooled_age18to24 after filter:", p.shape)
    print("  africa rows :", int(p["africa"].sum()))
    print("  other rows  :", int((1 - p["africa"]).sum()))
    print("  unique country (africa) :", p.loc[p.africa == 1, "country_string"].nunique())
    print("  unique country (other)  :", p.loc[p.africa == 0, "country_string"].nunique())

    cc = prep_pooled_cross_country()
    print("pooled_cross_country after filter:", cc.shape)
    print("  africa :", int(cc["africa"].sum()))
    print("  other  :", int((1 - cc["africa"]).sum()))
