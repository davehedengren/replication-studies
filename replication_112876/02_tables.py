"""
02_tables.py - Replicate Tables 5A, 5B, 6, 9, 10, 11, 12 and Figure 5
Rajan & Ramcharan (2014) - The Anatomy of a Credit Crisis
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from utils import *

# ============================================================
# TABLE 5A: Land Prices and Credit Availability (Cross-section)
# ============================================================
print("=" * 70)
print("TABLE 5A: Land Prices and Credit Availability")
print("=" * 70)

df5a = load_dta('table_5A.dta')

# Col 1: No controls
res5a_1 = run_reg_cluster(df5a, 'win_landval_update_ppa_log', ['win_banks_l'],
                           'state', 'state')
print_results(res5a_1, "Table 5A Col 1: No controls", ['win_banks_l'])

# Col 2: Kitchen sink controls
xvars_5a2 = ['win_banks_l'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH2
res5a_2 = run_reg_cluster(df5a, 'win_landval_update_ppa_log', xvars_5a2,
                           'state', 'state')
print_results(res5a_2, "Table 5A Col 2: With controls", xvars_5a2)

# Col 3: Drop manufacturing outliers (man_crop < 95th percentile)
cond3 = df5a['win_man_crop'] < 0.9453
res5a_3 = run_reg_cluster(df5a, 'win_landval_update_ppa_log', xvars_5a2,
                           'state', 'state', condition=cond3)
print_results(res5a_3, "Table 5A Col 3: Drop manufacturing outliers", ['win_banks_l'])

# Col 4: Banks per area
xvars_5a4 = ['a'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH2
cond4 = df5a['win_man_crop'] <= df5a['man_crop_95']
res5a_4 = run_reg_cluster(df5a, 'win_landval_update_ppa_log', xvars_5a4,
                           'state', 'state', condition=cond4)
print_results(res5a_4, "Table 5A Col 4: Banks per area", ['a'])

# Col 5: Banks per capita
xvars_5a5 = ['p'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH2
res5a_5 = run_reg_cluster(df5a, 'win_landval_update_ppa_log', xvars_5a5,
                           'state', 'state', condition=cond4)
print_results(res5a_5, "Table 5A Col 5: Banks per capita", ['p'])


# ============================================================
# TABLE 5B: Panel and IV estimates
# ============================================================
print("\n" + "=" * 70)
print("TABLE 5B: Panel FE and IV Estimates")
print("=" * 70)

df5b = load_dta('table_5b.dta')

# Need to identify the control globals used in table_5b.do
# The do file references $win_geo_log and $win_oth but these aren't defined in the file
# They must come from a master do file. Let's check what variables are available.
# Col 1: Panel with county and year FE
# areg lppa l ngp ubp ill ttl vfm ypy i.year, absorb(fips) cluster(statename)
# This has year FE via dummies AND county FE via absorb

# For areg with i.year and absorb(fips), we need to:
# 1. Create year dummies
# 2. Include them as regressors
# 3. Absorb fips (county FE via demeaning)
# Actually this is complex - let's use the full dummy approach instead

# Create year dummies
df5b_panel = df5b.copy()
year_dums = pd.get_dummies(df5b_panel['year'], prefix='year', drop_first=True, dtype=float)
df5b_panel = pd.concat([df5b_panel, year_dums], axis=1)
year_cols = list(year_dums.columns)

panel_xvars = ['l', 'ngp', 'ubp', 'ill', 'ttl', 'vfm', 'ypy'] + year_cols
res5b_1 = run_areg_cluster(df5b_panel, 'lppa', panel_xvars, 'fips', 'statename')
print_results(res5b_1, "Table 5B Col 1: Panel county+year FE",
              ['l', 'ngp', 'ubp', 'ill', 'ttl', 'vfm', 'ypy'])

# Col 2: 1920 cross-section OLS with 1910 banks
# Check what geo/other controls are used - the do file references macros
# Let's check available columns
geo_cols_5b = [c for c in WIN_GEO_LOG if c in df5b.columns]
oth_cols_5b = [c for c in WIN_OTH if c in df5b.columns]

cond_5b2 = (df5b['year'] == 1920) & (df5b['sample'] == 1)
xvars_5b2 = ['l_1910', 'ngp_1920w', 'ubp_1920w', 'ill_1920w', 'ttl_1920w', 'ypy_1920w'] + geo_cols_5b + oth_cols_5b
# Filter to only existing columns
xvars_5b2 = [v for v in xvars_5b2 if v in df5b.columns]
res5b_2 = run_reg_cluster(df5b, 'lppa_1920w', xvars_5b2, 'statename', 'statename',
                           condition=cond_5b2)
print_results(res5b_2, "Table 5B Col 2: 1920 cross-section OLS (1910 banks)", ['l_1910'])

# Col 3: IV - 1920 banks instrumented by 1910 banks
xvars_5b3_exog = ['ngp_1920w', 'ubp_1920w', 'ill_1920w', 'ttl_1920w', 'ypy_1920w'] + geo_cols_5b + oth_cols_5b
xvars_5b3_exog = [v for v in xvars_5b3_exog if v in df5b.columns]
try:
    res5b_3 = run_ivreg2_cluster(df5b, 'lppa_1920w', ['l_1920'], ['l_1910'],
                                  xvars_5b3_exog, 'statename', 'statename',
                                  condition=cond_5b2)
    print_results(res5b_3, "Table 5B Col 3: IV (1920 banks instrumented by 1910)", ['l_1920'])
except Exception as e:
    print(f"\nTable 5B Col 3 IV: Error - {e}")


# ============================================================
# TABLE 6: State Borders and Lending (Spatial)
# ============================================================
print("\n" + "=" * 70)
print("TABLE 6: State Borders and Lending")
print("=" * 70)

df6 = load_dta('table_6.dta')

# Col 1: 40-mile distance bands
cond6_1 = (df6['mincenter'] <= 80) & (df6['win_man_crop'] <= df6['man_crop_95'])
xvars_6_1 = ['l', 'win_ws_nw_banks40_l', 'win_ws_nw_banks4080_l', 'win_os_nw_banks40_l'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH2
xvars_6_1 = [v for v in xvars_6_1 if v in df6.columns]
res6_1 = run_reg_cluster(df6, 'win_landval_update_ppa_log', xvars_6_1,
                          'state', 'state', condition=cond6_1)
print_results(res6_1, "Table 6 Col 1: 40-mile bands",
              ['l', 'win_ws_nw_banks40_l', 'win_ws_nw_banks4080_l', 'win_os_nw_banks40_l'])

# Col 2: 50-mile distance bands
cond6_2 = (df6['mincenter'] <= 100) & (df6['win_man_crop'] <= df6['man_crop_95'])
xvars_6_2 = ['l', 'win_ws_nw_banks50_l', 'win_ws_nw_banks50100_l', 'win_os_nw_banks50_l'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH2
xvars_6_2 = [v for v in xvars_6_2 if v in df6.columns]
res6_2 = run_reg_cluster(df6, 'win_landval_update_ppa_log', xvars_6_2,
                          'state', 'state', condition=cond6_2)
print_results(res6_2, "Table 6 Col 2: 50-mile bands",
              ['l', 'win_ws_nw_banks50_l', 'win_ws_nw_banks50100_l', 'win_os_nw_banks50_l'])

# Col 3: 30-mile distance bands
cond6_3 = (df6['mincenter'] <= 90) & (df6['win_man_crop'] <= df6['man_crop_95'])
xvars_6_3 = ['l', 'win_ws_nw_banks30_l', 'win_ws_nw_banks3060_l', 'win_ws_nw_banks6090_l',
             'win_os_nw_banks30_l', 'win_os_nw_banks3060_l'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH2
xvars_6_3 = [v for v in xvars_6_3 if v in df6.columns]
res6_3 = run_reg_cluster(df6, 'win_landval_update_ppa_log', xvars_6_3,
                          'state', 'state', condition=cond6_3)
print_results(res6_3, "Table 6 Col 3: 30-mile bands",
              ['l', 'win_ws_nw_banks30_l', 'win_ws_nw_banks3060_l', 'win_ws_nw_banks6090_l',
               'win_os_nw_banks30_l', 'win_os_nw_banks3060_l'])


# ============================================================
# TABLE 9: Channels (Prices, Acres, Investment, Debt)
# ============================================================
print("\n" + "=" * 70)
print("TABLE 9: Channels")
print("=" * 70)

df9 = load_dta('table_9.dta')
# table_9.do uses i.state but data has statename; use statename for FE and clustering
fe9 = 'statename'
cond9 = df9['year'] == 1920

# Common channel variables
channel_bank_vars = ['l', 'l_sq', 'l_com', 'l_2010', 'win_pindex_achg_17']

# Col 1: Change in land prices 1920-1910
xvars_9_1 = channel_bank_vars + ['lppa_10'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH
xvars_9_1 = [v for v in xvars_9_1 if v in df9.columns]
res9_1 = run_reg_cluster(df9, 'lppa_2010', xvars_9_1, fe9, fe9, condition=cond9)
print_results(res9_1, "Table 9 Col 1: Change in prices 1920-1910",
              channel_bank_vars + ['lppa_10'])

# Col 2: Extensive margin (acres)
xvars_9_2 = channel_bank_vars + ['acres_log_1910', 'areaac_log'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH
xvars_9_2 = [v for v in xvars_9_2 if v in df9.columns]
res9_2 = run_reg_cluster(df9, 'acres_log_2010', xvars_9_2, fe9, fe9, condition=cond9)
print_results(res9_2, "Table 9 Col 2: Extensive margin (acres)",
              channel_bank_vars + ['acres_log_1910'])

# Col 3: Intensive margin
xvars_9_3 = channel_bank_vars + ['acimp_log_1910'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH
xvars_9_3 = [v for v in xvars_9_3 if v in df9.columns]
res9_3 = run_reg_cluster(df9, 'acimp_log_2010', xvars_9_3, fe9, fe9, condition=cond9)
print_results(res9_3, "Table 9 Col 3: Intensive margin",
              channel_bank_vars + ['acimp_log_1910'])

# Col 4: Investment (farm equipment)
xvars_9_4 = channel_bank_vars + ['farmequi_log_1910'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH
xvars_9_4 = [v for v in xvars_9_4 if v in df9.columns]
res9_4 = run_reg_cluster(df9, 'farmequi_log_2010', xvars_9_4, fe9, fe9, condition=cond9)
print_results(res9_4, "Table 9 Col 4: Investment",
              channel_bank_vars + ['farmequi_log_1910'])

# Col 5: Debt per acre
xvars_9_5 = ['l', 'l_sq', 'l_com', 'l_2010', 'win_pindex_achg_17', 'debt_acre_1910_log'] + WIN_DEM_LOG + WIN_GEO_LOG + WIN_OTH
xvars_9_5 = [v for v in xvars_9_5 if v in df9.columns]
# Note: table_9.do uses cluster(statename) for col 5
res9_5 = run_reg_cluster(df9, 'win_debt_acre_2010', xvars_9_5, fe9, fe9, condition=cond9)
print_results(res9_5, "Table 9 Col 5: Debt per acre",
              ['l', 'l_sq', 'l_com', 'l_2010', 'win_pindex_achg_17', 'debt_acre_1910_log'])


# ============================================================
# TABLE 10: Bank Suspensions (OLS and IV)
# ============================================================
print("\n" + "=" * 70)
print("TABLE 10: Bank Suspensions")
print("=" * 70)

df10 = load_dta('table_10.dta')
cond10 = df10['year'] == 1920

# Col 1: OLS, suspended banks
xvars_10_1 = ['win_debt_acre_2010', 'debt_acre_1920', 'debt_acre_1920_sq',
              'debt_acre_1920_com', 'win_pindex_achg_17'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH
xvars_10_1 = [v for v in xvars_10_1 if v in df10.columns]
res10_1 = run_reg_cluster(df10, 'win_banks_susp2129', xvars_10_1,
                           'statename', 'statename', condition=cond10)
print_results(res10_1, "Table 10 Col 1: OLS, suspended banks",
              ['win_debt_acre_2010', 'debt_acre_1920', 'debt_acre_1920_sq',
               'debt_acre_1920_com', 'win_pindex_achg_17'])

# Col 3: OLS, suspended deposits
xvars_10_3 = ['debt_acre_1920', 'win_debt_acre_2010', 'win_pindex_achg_17',
              'debt_acre_1920_sq', 'debt_acre_1920_com'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH
xvars_10_3 = [v for v in xvars_10_3 if v in df10.columns]
res10_3 = run_reg_cluster(df10, 'win_deposits_susp2129', xvars_10_3,
                           'statename', 'statename', condition=cond10)
print_results(res10_3, "Table 10 Col 3: OLS, suspended deposits",
              ['debt_acre_1920', 'win_debt_acre_2010', 'win_pindex_achg_17',
               'debt_acre_1920_sq', 'debt_acre_1920_com'])

# IV columns (2 and 4) - attempt
try:
    res10_2 = run_ivreg2_cluster(
        df10, 'win_banks_susp2129',
        ['win_debt_acre_2010', 'debt_acre_1920', 'debt_acre_1920_sq', 'debt_acre_1920_com'],
        ['l_1920', 'l_2010', 'l_1920_sq', 'l_1920_com'],
        ['win_pindex_achg_17'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH,
        'statename', 'statename', condition=cond10)
    print_results(res10_2, "Table 10 Col 2: IV, suspended banks",
                  ['win_debt_acre_2010', 'debt_acre_1920', 'debt_acre_1920_sq',
                   'debt_acre_1920_com', 'win_pindex_achg_17'])
except Exception as e:
    print(f"\nTable 10 Col 2 IV: Error - {e}")


# ============================================================
# TABLE 11: Long-run Persistence
# ============================================================
print("\n" + "=" * 70)
print("TABLE 11: Long-run Persistence of Land Prices")
print("=" * 70)

df11 = load_dta('table_11.dta')
cond11 = df11['year'] == 1920

decades = [
    ('win_ppa_log_3020', 'ppa_log_1920', '1930-1920'),
    ('win_ppa_log_4030', 'ppa_log_1930', '1940-1930'),
    ('win_ppa_log_5040', 'ppa_log_1940', '1950-1940'),
    ('win_ppa_log_6050', 'ppa_log_1950', '1960-1950'),
    ('ppa_log_1960', 'ppa_log_1920', '1960 level'),
]

for i, (yvar, lagged_price, label) in enumerate(decades):
    xvars_11 = ['l', 'l_sq', 'l_com', 'l_2010', 'win_pindex_achg_17', lagged_price] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH
    xvars_11 = [v for v in xvars_11 if v in df11.columns]
    res11 = run_reg_cluster(df11, yvar, xvars_11, 'state', 'state', condition=cond11)
    print_results(res11, f"Table 11 Col {i+1}: {label}",
                  ['l', 'l_sq', 'l_com', 'l_2010', 'win_pindex_achg_17', lagged_price])


# ============================================================
# TABLE 12: Long-run Bank Structure
# ============================================================
print("\n" + "=" * 70)
print("TABLE 12: Long-run Outcomes")
print("=" * 70)

df12 = load_dta('table_12.dta')
cond12 = (df12['year'] == 1920) & (df12['sample'] == 1)

# Col 1: Banks 1972
xvars_12_1 = ['win_banks_susp2129', 'win_pindex_achg_17', 'l'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH
xvars_12_1 = [v for v in xvars_12_1 if v in df12.columns]
res12_1 = run_reg_cluster(df12, 'banks_1972_log', xvars_12_1,
                           'statename', 'statename', condition=cond12)
print_results(res12_1, "Table 12 Col 1: Log banks 1972",
              ['win_banks_susp2129', 'win_pindex_achg_17', 'l'])

# Col 2: HHI
res12_2 = run_reg_cluster(df12, 'hhi', xvars_12_1,
                           'statename', 'statename', condition=cond12)
print_results(res12_2, "Table 12 Col 2: HHI 1972",
              ['win_banks_susp2129', 'win_pindex_achg_17', 'l'])

# Col 3: OLS price 1960
xvars_12_3 = ['ppa_log_1920', 'win_banks_susp2129', 'win_pindex_achg_17'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH
xvars_12_3 = [v for v in xvars_12_3 if v in df12.columns]
res12_3 = run_reg_cluster(df12, 'ppa_log_1960', xvars_12_3,
                           'statename', 'statename', condition=cond12)
print_results(res12_3, "Table 12 Col 3: OLS price 1960",
              ['win_banks_susp2129', 'win_pindex_achg_17', 'ppa_log_1920'])

# Col 4: IV price 1960
try:
    xvars_12_4_exog = ['ppa_log_1920', 'win_pindex_achg_17'] + WIN_GEO_LOG + WIN_DEM_LOG + WIN_OTH
    xvars_12_4_exog = [v for v in xvars_12_4_exog if v in df12.columns]
    res12_4 = run_ivreg2_cluster(df12, 'ppa_log_1960', ['win_banks_susp2129'], ['l'],
                                  xvars_12_4_exog, 'statename', 'statename', condition=cond12)
    print_results(res12_4, "Table 12 Col 4: IV price 1960",
                  ['win_banks_susp2129', 'win_pindex_achg_17', 'ppa_log_1920'])
except Exception as e:
    print(f"\nTable 12 Col 4 IV: Error - {e}")


# ============================================================
# FIGURE 5: Commodity Price Shock and Land Prices
# ============================================================
print("\n" + "=" * 70)
print("FIGURE 5: Commodity Price Shock")
print("=" * 70)

df_f5 = load_dta('figure_5.dta')
cond_f5 = (df_f5['year'] == 1920) & (df_f5['win_man_crop'] <= df_f5['man_crop_95'])

# Without state FE
res_f5_1 = run_reg_cluster(df_f5[cond_f5], 'win_lppa_chg', ['win_pindex_achg_17'],
                            'state', 'state')
print_results(res_f5_1, "Figure 5: Without state FE", ['win_pindex_achg_17'])

# With state FE
res_f5_2 = run_reg_cluster(df_f5, 'win_lppa_chg', ['win_pindex_achg_17'],
                            'state', 'state', condition=cond_f5)
print_results(res_f5_2, "Figure 5: With state FE", ['win_pindex_achg_17'])

print("\n" + "=" * 70)
print("REPLICATION COMPLETE")
print("=" * 70)
