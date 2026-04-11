"""Build the station-day and station-month panels that feed Table 1.
Mirrors steps 1-3 of Prepare_Data.do."""

import numpy as np
import pandas as pd
from utils import DATA, OUT, WEATHER_VARS, build_station_day, load_dta, \
    DEADLINE_WAVE1, DEADLINE_WAVE2


def main():
    print("=== 01_clean: building station-day panel ===")
    sd = build_station_day()
    print(f"station_day rows: {len(sd):,}  cities: {sd.code_city.nunique()}  "
          f"stations: {sd.pm10_n.nunique()}")
    sd.to_parquet(OUT / "station_day_1116.parquet")

    # The RD setup needs T = date - auto_date and month-of-year
    sd["T"] = sd["date"] - sd["auto_date"]
    sd["month"] = pd.to_datetime(sd["date"], unit="D", origin="1960-01-01").dt.month
    sd["year"] = pd.to_datetime(sd["date"], unit="D", origin="1960-01-01").dt.year
    sd.to_parquet(OUT / "station_day_rd.parquet")

    # Deadline matched sample — Stata does a geonear 1:N distance match on
    # auto_date==19359 vs 19724 (400 km radius). We approximate with a fixed
    # haversine-based match giving, for each Wave-1 deadline station, the set of
    # Wave-2 deadline stations within 400 km. We reproduce both sides exactly.
    stations = load_dta("station_list")
    w1 = stations[stations["auto_date"] == DEADLINE_WAVE1].copy()
    w2 = stations[stations["auto_date"] == DEADLINE_WAVE2].copy()

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1r, lat2r = np.radians(lat1), np.radians(lat2)
        dlat = lat2r - lat1r
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    # Build the (w1,w2) pairs within 400 km like geonear
    lat1 = w1["station_lat"].to_numpy()[:, None]
    lon1 = w1["station_lon"].to_numpy()[:, None]
    lat2 = w2["station_lat"].to_numpy()[None, :]
    lon2 = w2["station_lon"].to_numpy()[None, :]
    D = haversine(lat1, lon1, lat2, lon2)  # km
    # for each w1 row find nearest w2 (geonear n=1 by default)
    nearest = D.argmin(axis=1)
    km = D[np.arange(len(w1)), nearest]
    pairs = pd.DataFrame({
        "pm10_n1": w1["pm10_n"].values,          # wave-1 station
        "pm10_n": w2["pm10_n"].values[nearest],  # matched wave-2 station
        "km": km,
    })
    pairs = pairs[pairs["km"] <= 400].reset_index(drop=True)
    print(f"matched pairs (<=400km): {len(pairs)}  (Stata had 123 W1 stations)")

    treated = pairs.rename(columns={"pm10_n1": "pm10_n",
                                    "pm10_n": "_drop"})[["pm10_n"]].assign(treat=1)
    # control side: each wave-2 station that was matched at least once (with
    # replacement)
    control = pairs[["pm10_n"]].assign(treat=0)

    sd12 = sd[(sd["year"].isin([2012, 2013]))].copy()
    treated_panel = sd12.merge(treated, on="pm10_n", how="inner")
    control_panel = sd12.merge(control, on="pm10_n", how="inner")
    did = pd.concat([treated_panel, control_panel], ignore_index=True)
    did = did.drop(columns=["so2", "no2", "phase", "treat"], errors="ignore")
    with np.errstate(divide="ignore"):
        did["l_pm10"] = np.log(did["pm10"].replace(0, np.nan))
    did.to_parquet(OUT / "did_ddl_match.parquet")
    print(f"did_ddl_match rows: {len(did):,}")

    # Station-month panel (Table 1 Panel A monthly AOD and bandwidth)
    print("=== building station_month ===")
    sm = sd.copy()
    sm["n_month"] = (
        pd.to_datetime(sm["date"], unit="D", origin="1960-01-01").dt.to_period("M").astype("int64")
        - pd.to_datetime(sm["auto_date"], unit="D", origin="1960-01-01").dt.to_period("M").astype("int64")
    )
    sm["after"] = (sm["n_month"] >= 0).astype(int)
    for v in ["pm10", "so2", "no2"] + WEATHER_VARS:
        sm[v] = sm.groupby(["pm10_n", "year", "month"])[v].transform("mean")
    sm = sm.drop_duplicates(["pm10_n", "year", "month"]).copy()
    sm = sm.drop(columns=["date"])
    aod = load_dta("aod_month")
    sm = sm.merge(aod, on=["pm10_n", "year", "month"], how="inner")
    sm.to_parquet(OUT / "station_month.parquet")
    print(f"station_month rows: {len(sm):,}")

    # mask/filter search city-month (for Table 3)
    print("=== building search_city_month ===")
    sdc = sd.copy()
    for v in ["pm10", "so2", "no2"] + WEATHER_VARS:
        sdc[v] = sdc.groupby(["code_city", "date"])[v].transform("mean")
    sdc = sdc.sort_values(["code_city", "date", "auto_date"], ascending=[True, True, False])
    sdc = sdc.drop_duplicates(["code_city", "date"]).copy()
    sdc = sdc.drop(columns=["pm10_n"])
    mfs = load_dta("mask_filter_search")
    sdc = sdc.merge(mfs, on=["code_city", "date"], how="inner")
    sdc["year"] = pd.to_datetime(sdc["date"], unit="D", origin="1960-01-01").dt.year
    sdc["month"] = pd.to_datetime(sdc["date"], unit="D", origin="1960-01-01").dt.month
    sdc["n_month"] = (
        pd.to_datetime(sdc["date"], unit="D", origin="1960-01-01").dt.to_period("M").astype("int64")
        - pd.to_datetime(sdc["auto_date"], unit="D", origin="1960-01-01").dt.to_period("M").astype("int64")
    )
    for v in ["pm10", "so2", "no2", "pmmask", "filter"] + WEATHER_VARS:
        sdc[v] = sdc.groupby(["code_city", "n_month"])[v].transform("mean")
    sdc = sdc.drop_duplicates(["code_city", "n_month"]).copy()
    for v in ["pmmask", "filter"]:
        sdc[f"l_{v}"] = np.log1p(sdc[v])
    sdc.to_parquet(OUT / "search_city_month.parquet")
    print(f"search_city_month rows: {len(sdc):,}")


if __name__ == "__main__":
    main()
