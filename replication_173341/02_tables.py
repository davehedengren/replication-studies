"""
Phase 2: Replicate Main Tables 1-5 for 173341.
"""
import sys
sys.path.insert(0, '.')
from replication_173341.utils import *


def table1():
    """Table 1: Interactions with Politicians (Summary Statistics + Rainfall Correlations)."""
    print("\n" + "=" * 70)
    print("TABLE 1: Interactions with Politicians")
    print("=" * 70)

    indiv = load_individual()
    # Rename to match Stata code
    indiv.rename(columns={'c_ask_private_pol_postcoded': 'c_ask_private_pol_post'}, inplace=True)

    summary_list_1 = [
        'frequent_interactor', 'vote_same_party', 'vote_together',
        'c_candidate_visit', 'c_declared_dummy', 'declare_body',
        'declare_house', 'declare_rally'
    ]
    summary_list_2 = [
        'c_ask_private_pol_post', 'ask_pol_private_2013',
        'c_pol_askrec_priv_post', 'askrec_pol_private_2013'
    ]
    all_vars = summary_list_1 + summary_list_2

    # Published values for comparison
    published = {
        'c_ask_private_pol_post':  {'mean': 0.195, 'sd': 0.397, 'beta': -0.032, 'se': 0.010},
        'ask_pol_private_2013':    {'mean': 0.086, 'sd': 0.281, 'beta': -0.003, 'se': 0.007},
        'c_pol_askrec_priv_post':  {'mean': 0.115, 'sd': 0.319, 'beta': -0.020, 'se': 0.008},
        'askrec_pol_private_2013': {'mean': 0.039, 'sd': 0.193, 'beta': -0.001, 'se': 0.005},
        'frequent_interactor':     {'mean': 0.184, 'sd': 0.387, 'beta': -0.010, 'se': 0.008},
        'c_candidate_visit':       {'mean': 0.696, 'sd': 0.460, 'beta': 0.015, 'se': 0.011},
        'vote_same_party':         {'mean': 0.718, 'sd': 0.450, 'beta': -0.019, 'se': 0.015},
        'vote_together':           {'mean': 0.773, 'sd': 0.419, 'beta': -0.023, 'se': 0.011},
        'c_declared_dummy':        {'mean': 0.485, 'sd': 0.500, 'beta': -0.066, 'se': 0.016},
        'declare_body':            {'mean': 0.185, 'sd': 0.388, 'beta': -0.021, 'se': 0.009},
        'declare_house':           {'mean': 0.387, 'sd': 0.487, 'beta': -0.059, 'se': 0.016},
        'declare_rally':           {'mean': 0.218, 'sd': 0.413, 'beta': -0.036, 'se': 0.010},
    }

    # Order as in the tex file
    var_order = [
        'c_ask_private_pol_post', 'ask_pol_private_2013',
        'c_pol_askrec_priv_post', 'askrec_pol_private_2013',
        'frequent_interactor', 'c_candidate_visit',
        'vote_same_party', 'vote_together',
        'c_declared_dummy', 'declare_body', 'declare_house', 'declare_rally'
    ]

    var_labels = {
        'c_ask_private_pol_post': 'Request Private Good, 2012',
        'ask_pol_private_2013': 'Request Private Good, 2013',
        'c_pol_askrec_priv_post': 'Request & Receive Private Good, 2012',
        'askrec_pol_private_2013': 'Request & Receive Private Good, 2013',
        'frequent_interactor': 'Frequent Interactions w/ Politician',
        'c_candidate_visit': 'Received Visit from Candidate Rep',
        'vote_same_party': 'Voted Same Coalition',
        'vote_together': 'All HH Members Voting Same',
        'c_declared_dummy': 'Any Declared Support',
        'declare_body': 'Declared Support on Body',
        'declare_house': 'Declared Support on House',
        'declare_rally': 'Declared Support at Rally',
    }

    results = []
    print(f"\n{'Variable':<42} {'Mean':>7} {'[SD]':>9} {'Beta':>8} {'(SE)':>8} {'Match?':>7}")
    print("-" * 85)

    for v in var_order:
        # Mean and SD
        mean_v = indiv[v].mean()
        sd_v = indiv[v].std()

        # Regression on rainfall
        if v in ['ask_pol_private_2013', 'askrec_pol_private_2013']:
            rain = 'rainfall_2013_std'
        else:
            rain = 'rainfall_2012_std'

        mask = indiv[v].notna() & indiv[rain].notna() & indiv['b_clusters'].notna()
        y = indiv.loc[mask, v].astype(float)
        X = indiv.loc[mask, [rain]].astype(float)
        cl = indiv.loc[mask, 'b_clusters']
        model = sm.OLS(y, sm.add_constant(X)).fit(
            cov_type='cluster', cov_kwds={'groups': cl}
        )
        beta = model.params[rain]
        se = model.bse[rain]

        pub = published[v]
        mean_match = abs(mean_v - pub['mean']) < 0.0015
        beta_match = abs(beta - pub['beta']) < 0.0015
        se_match = abs(se - pub['se']) < 0.0015
        all_match = mean_match and beta_match and se_match

        print(f"{var_labels[v]:<42} {mean_v:7.3f} [{sd_v:.3f}] {beta:8.3f} ({se:.3f}) {'✓' if all_match else '✗'}")

        results.append({
            'variable': v,
            'label': var_labels[v],
            'mean': mean_v, 'sd': sd_v,
            'beta': beta, 'se': se,
            'pub_mean': pub['mean'], 'pub_beta': pub['beta'], 'pub_se': pub['se'],
            'match': all_match
        })

    n_match = sum(1 for r in results if r['match'])
    print(f"\nMatched: {n_match}/{len(results)} variables")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'table1.csv'), index=False)
    return results


def table2():
    """Table 2: Cisterns Treatment and Vulnerability."""
    print("\n" + "=" * 70)
    print("TABLE 2: Cisterns Treatment, Rainfall Shocks, and Vulnerability")
    print("=" * 70)

    hh = load_household()

    # Published values
    pub_panelA = {
        'd_Happiness':           {'beta': 0.092, 'se': 0.037},
        'd_Health':              {'beta': 0.075, 'se': 0.033},
        'd_Child_Food_Security': {'beta': 0.084, 'se': 0.054},
        'd_Overall_index':       {'beta': 0.126, 'se': 0.043},
    }
    pub_panelB = {
        'd_Happiness':           {'beta': 0.046, 'se': 0.016, 'N': 1128, 'mean_treat': 3.377, 'mean_ctrl': 3.289, 'sd_ctrl': 0.646},
        'd_Health':              {'beta': 0.039, 'se': 0.017, 'N': 1052, 'mean_treat': 2.868, 'mean_ctrl': 2.791, 'sd_ctrl': 0.535},
        'd_Child_Food_Security': {'beta': 0.046, 'se': 0.026, 'N': 1128, 'mean_treat': -0.277, 'mean_ctrl': -0.340, 'sd_ctrl': 0.990},
        'd_Overall_index':       {'beta': 0.064, 'se': 0.019, 'N': 1128, 'mean_treat': 0.061, 'mean_ctrl': -0.056, 'sd_ctrl': 0.714},
        'a_Household_Expenditure': {'beta': 24.736, 'se': 6.657, 'N': 1281, 'mean_treat': 364.467, 'mean_ctrl': 370.877, 'sd_ctrl': 207.744},
    }

    depvars_A = ['d_Happiness', 'd_Health', 'd_Child_Food_Security', 'd_Overall_index']
    depvars_B_2013 = ['d_Happiness', 'd_Health', 'd_Child_Food_Security', 'd_Overall_index']
    depvars_B_2011 = ['a_Household_Expenditure']

    labels = {
        'd_Happiness': '-(CES-D) Scale',
        'd_Health': 'SRHS Index',
        'd_Child_Food_Security': 'Child Food Security',
        'd_Overall_index': 'Overall Index',
        'a_Household_Expenditure': 'Total HH Expenditure',
    }

    # Panel A: reg depvar treatment i.mun_id, cluster(b_clusters)
    print("\n--- Panel A: Effect of Treatment on Vulnerability ---")
    print(f"{'Variable':<25} {'Beta':>8} {'SE':>8} {'Pub_B':>8} {'Pub_SE':>8} {'Match?':>7}")
    print("-" * 68)

    results_A = []
    for v in depvars_A:
        model, n = ols_with_fe(v, ['treatment'], 'mun_id', 'b_clusters', hh)
        beta = model.params['treatment']
        se_adj = get_adj_se(model)['treatment']
        pub = pub_panelA[v]
        match = abs(beta - pub['beta']) < 0.0015 and abs(se_adj - pub['se']) < 0.0015
        print(f"{labels[v]:<25} {beta:8.3f} {se_adj:8.3f} {pub['beta']:8.3f} {pub['se']:8.3f} {'✓' if match else '✗'}")
        results_A.append({'variable': v, 'beta': beta, 'se': se_adj, 'N': n, 'match': match})

    # Panel B: reg depvar rainfall_std, cluster(b_clusters) - NO FE
    print("\n--- Panel B: Effect of Rainfall on Vulnerability ---")
    print(f"{'Variable':<25} {'Beta':>8} {'SE':>8} {'N':>6} {'Pub_B':>8} {'Pub_SE':>8} {'Pub_N':>6} {'Match?':>7}")
    print("-" * 82)

    results_B = []
    for v in depvars_B_2013:
        rain = 'rainfall_2013_std'
        mask = hh[v].notna() & hh[rain].notna() & hh['b_clusters'].notna()
        y = hh.loc[mask, v].astype(float)
        X = hh.loc[mask, [rain]].astype(float)
        cl = hh.loc[mask, 'b_clusters']
        model = sm.OLS(y, sm.add_constant(X)).fit(
            cov_type='cluster', cov_kwds={'groups': cl}
        )
        beta = model.params[rain]
        se = model.bse[rain]
        n_obs = int(model.nobs)
        pub = pub_panelB[v]

        # Means for treatment and control
        sample = hh[mask]
        mean_treat = sample.loc[sample['treatment'] == 1, v].mean()
        mean_ctrl = sample.loc[sample['treatment'] == 0, v].mean()
        sd_ctrl = sample.loc[sample['treatment'] == 0, v].std()

        match = abs(beta - pub['beta']) < 0.0015 and abs(se - pub['se']) < 0.0015 and n_obs == pub['N']
        print(f"{labels[v]:<25} {beta:8.3f} {se:8.3f} {n_obs:6d} {pub['beta']:8.3f} {pub['se']:8.3f} {pub['N']:6d} {'✓' if match else '✗'}")
        results_B.append({
            'variable': v, 'beta': beta, 'se': se, 'N': n_obs,
            'mean_treat': mean_treat, 'mean_ctrl': mean_ctrl, 'sd_ctrl': sd_ctrl,
            'match': match
        })

    # 2011 expenditure
    for v in depvars_B_2011:
        rain = 'rainfall_2011_std'
        mask = hh[v].notna() & hh[rain].notna() & hh['b_clusters'].notna()
        y = hh.loc[mask, v].astype(float)
        X = hh.loc[mask, [rain]].astype(float)
        cl = hh.loc[mask, 'b_clusters']
        model = sm.OLS(y, sm.add_constant(X)).fit(
            cov_type='cluster', cov_kwds={'groups': cl}
        )
        beta = model.params[rain]
        se = model.bse[rain]
        n_obs = int(model.nobs)
        pub = pub_panelB[v]

        sample = hh[mask]
        mean_treat = sample.loc[sample['treatment'] == 1, v].mean()
        mean_ctrl = sample.loc[sample['treatment'] == 0, v].mean()
        sd_ctrl = sample.loc[sample['treatment'] == 0, v].std()

        match = abs(beta - pub['beta']) < 0.15 and abs(se - pub['se']) < 0.15 and n_obs == pub['N']
        print(f"{labels[v]:<25} {beta:8.3f} {se:8.3f} {n_obs:6d} {pub['beta']:8.3f} {pub['se']:8.3f} {pub['N']:6d} {'✓' if match else '✗'}")
        results_B.append({
            'variable': v, 'beta': beta, 'se': se, 'N': n_obs,
            'mean_treat': mean_treat, 'mean_ctrl': mean_ctrl, 'sd_ctrl': sd_ctrl,
            'match': match
        })

    return results_A, results_B


def table3():
    """Table 3: Citizen Requests, Cisterns Treatment, and Rainfall Shocks."""
    print("\n" + "=" * 70)
    print("TABLE 3: Citizen Requests, Cisterns Treatment, and Rainfall Shocks")
    print("=" * 70)

    stacked = load_stacked()

    # Create interaction variables (matching Stata code)
    stacked['treat_2012'] = stacked['treatment'] * stacked['year2012']
    stacked['treat_2013'] = stacked['treatment'] * stacked['year2013']
    stacked['rainfall_std_stacked_2012'] = stacked['rainfall_std_stacked'] * stacked['year2012']
    stacked['rainfall_std_stacked_2013'] = stacked['rainfall_std_stacked'] * stacked['year2013']
    stacked['treat_rain_std'] = stacked['treatment'] * stacked['rainfall_std_stacked']

    # Published values: coefficients and SEs for each column
    published = {
        1: {'treatment': (-0.028, 0.012)},
        2: {'rainfall_std_stacked': (-0.025, 0.010)},
        3: {'treatment': (-0.028, 0.012), 'rainfall_std_stacked': (-0.025, 0.010)},
        4: {'treatment': (-0.028, 0.012), 'rainfall_std_stacked': (-0.024, 0.011), 'treat_rain_std': (-0.002, 0.012)},
        5: {'treat_2012': (-0.026, 0.016), 'treat_2013': (-0.032, 0.016)},
        6: {'rainfall_std_stacked_2012': (-0.040, 0.012), 'rainfall_std_stacked_2013': (-0.009, 0.012)},
        7: {'treatment': (-0.028, 0.012), 'rainfall_std_stacked': (-0.016, 0.008)},
        8: {'treatment': (-0.004, 0.005), 'rainfall_std_stacked': (-0.006, 0.004)},
    }

    # Specifications
    specs = [
        # (col, depvar, regressors)
        (1, 'ask_private_stacked', ['treatment']),
        (2, 'ask_private_stacked', ['rainfall_std_stacked']),
        (3, 'ask_private_stacked', ['treatment', 'rainfall_std_stacked']),
        (4, 'ask_private_stacked', ['treatment', 'rainfall_std_stacked', 'treat_rain_std']),
        (5, 'ask_private_stacked', ['treat_2012', 'treat_2013']),
        (6, 'ask_private_stacked', ['rainfall_std_stacked_2012', 'rainfall_std_stacked_2013']),
        (7, 'ask_nowater_private_stacked', ['treatment', 'rainfall_std_stacked']),
        (8, 'ask_public_stacked', ['treatment', 'rainfall_std_stacked']),
    ]

    results = []
    for col, depvar, regressors in specs:
        all_X = regressors + ['year2012']
        model, n = ols_with_fe(depvar, all_X, 'mun_id', 'b_clusters', stacked)
        se_adj = get_adj_se(model)

        pub = published[col]
        print(f"\nColumn ({col}): {depvar}")
        print(f"  N = {n}")
        all_match = True
        for reg in regressors:
            beta = model.params[reg]
            se = se_adj[reg]
            if reg in pub:
                pb, ps = pub[reg]
                match = abs(beta - pb) < 0.0015 and abs(se - ps) < 0.0015
                print(f"  {reg:<35} {beta:8.3f} ({se:.3f})  pub: {pb:8.3f} ({ps:.3f}) {'✓' if match else '✗'}")
                if not match:
                    all_match = False
            else:
                print(f"  {reg:<35} {beta:8.3f} ({se:.3f})")

        # Means
        cols_needed = list(dict.fromkeys([depvar] + all_X + ['mun_id', 'b_clusters', 'treatment']))
        sample = stacked[cols_needed].dropna()
        mean_treat = sample.loc[sample['treatment'] == 1, depvar].mean()
        mean_ctrl = sample.loc[sample['treatment'] == 0, depvar].mean()
        print(f"  Mean Y (Treatment): {mean_treat:.3f}, Mean Y (Control): {mean_ctrl:.3f}")

        results.append({
            'column': col, 'depvar': depvar, 'N': n,
            'all_match': all_match, 'model': model
        })

    # Homogeneity tests
    # Col 5: test treat_2012 = treat_2013
    model5 = results[4]['model']
    idx_2012 = list(model5.params.index).index('treat_2012')
    idx_2013 = list(model5.params.index).index('treat_2013')
    r_matrix = np.zeros((1, len(model5.params)))
    r_matrix[0, idx_2012] = 1
    r_matrix[0, idx_2013] = -1
    f_test = model5.f_test(r_matrix)
    p_treat = float(f_test.pvalue)
    print(f"\nHomogeneity test (treat_2012 = treat_2013): p = {p_treat:.3f} (published: 0.749)")

    # Col 6: test rainfall_2012 = rainfall_2013
    model6 = results[5]['model']
    idx_r2012 = list(model6.params.index).index('rainfall_std_stacked_2012')
    idx_r2013 = list(model6.params.index).index('rainfall_std_stacked_2013')
    r_matrix = np.zeros((1, len(model6.params)))
    r_matrix[0, idx_r2012] = 1
    r_matrix[0, idx_r2013] = -1
    f_test = model6.f_test(r_matrix)
    p_rain = float(f_test.pvalue)
    print(f"Homogeneity test (rain_2012 = rain_2013):   p = {p_rain:.3f} (published: 0.033)")

    return results


def table4():
    """Table 4: Cisterns Treatment and Electoral Outcomes (2012)."""
    print("\n" + "=" * 70)
    print("TABLE 4: Cisterns Treatment and Electoral Outcomes (2012)")
    print("=" * 70)

    # Published values: (beta, se, bootstrap_p)
    published = {
        ('incumbent_votes_section', 21): {'b1': -0.103, 'se1': 0.058, 'bt1': 0.038, 'b2': 0.022, 'se2': 0.044, 'bt2': 0.517, 'N': 909},
        ('incumbent_votes_section', 39): {'b1': -0.078, 'se1': 0.049, 'bt1': 0.090, 'b2': 0.036, 'se2': 0.038, 'bt2': 0.316, 'N': 1641},
        ('challenger_votes_section', 21): {'b1': 0.100, 'se1': 0.073, 'bt1': 0.088, 'b2': -0.034, 'se2': 0.058, 'bt2': 0.458, 'N': 909},
        ('turnout', 21): {'b1': -0.010, 'se1': 0.059, 'bt1': 0.834, 'b2': -0.001, 'se2': 0.049, 'bt2': 0.993, 'N': 909},
        ('blank_null', 21): {'b1': -0.008, 'se1': 0.031, 'bt1': 0.829, 'b2': 0.011, 'se2': 0.020, 'bt2': 0.576, 'N': 909},
    }

    depvar_list = ['incumbent_votes_section', 'incumbent_votes_section',
                   'challenger_votes_section', 'turnout', 'blank_null']
    samples = [21, 39, 21, 21, 21]

    col_labels = {
        ('incumbent_votes_section', 21): 'Votes for Incumbent Mayor (1)',
        ('incumbent_votes_section', 39): 'Votes for Incumbent Group (2)',
        ('challenger_votes_section', 21): 'Votes for Challenger (3)',
        ('turnout', 21): 'Turnout (4)',
        ('blank_null', 21): 'Blank and Null (5)',
    }

    print(f"\n{'Column':<35} {'b1':>8} {'se1':>8} {'b2':>8} {'se2':>8} {'N':>6} {'Match?':>7}")
    print("-" * 80)

    for depvar, sample in zip(depvar_list, samples):
        voting = load_voting()

        if sample == 39:
            mask = (
                ((voting['name_match'] == 1) | (voting['vp_match'] == 1) |
                 (voting['pty_match'] == 1) | (voting['coalition_match'] == 1)) &
                (voting['eligible_in_2008'].notna())
            )
            voting = voting[mask].copy()
            voting.rename(columns={
                'tot_treat_by_section_2_39': 'tot_treat_by_section_2',
                'tot_study_2_39': 'tot_study_2'
            }, inplace=True)
        else:
            mask = (voting['name_match'] == 1) & (voting['eligible_in_2008'].notna())
            voting = voting[mask].copy()
            voting.rename(columns={
                'tot_treat_by_section_2_21': 'tot_treat_by_section_2',
                'tot_study_2_21': 'tot_study_2'
            }, inplace=True)

        X_vars = ['tot_treat_by_section_2', 'tot_study_2', 'eligible']

        # reghdfe with location_id absorbed, cluster on location_id
        model, n = reghdfe_absorb(depvar, X_vars, 'location_id', 'location_id', voting)

        b1 = model.params['tot_treat_by_section_2']
        b2 = model.params['tot_study_2']

        # Apply reghdfe DOF adjustment to SEs
        se1 = model.bse['tot_treat_by_section_2'] * model._se_adj
        se2 = model.bse['tot_study_2'] * model._se_adj

        key = (depvar, sample)
        pub = published.get(key)
        if pub:
            match_b1 = abs(b1 - pub['b1']) < 0.0015
            match_se1 = abs(se1 - pub['se1']) < 0.0015
            match_b2 = abs(b2 - pub['b2']) < 0.0015
            match_se2 = abs(se2 - pub['se2']) < 0.0015
            match_n = n == pub['N']
            all_match = match_b1 and match_se1 and match_b2 and match_se2 and match_n
            label = col_labels.get(key, f"{depvar} ({sample})")
            print(f"{label:<35} {b1:8.3f} {se1:8.3f} {b2:8.3f} {se2:8.3f} {n:6d} {'✓' if all_match else '✗'}")
            if not all_match:
                print(f"  Published:                        {pub['b1']:8.3f} {pub['se1']:8.3f} {pub['b2']:8.3f} {pub['se2']:8.3f} {pub['N']:6d}")
        else:
            print(f"{depvar} ({sample}): b1={b1:.3f} se1={se1:.3f} b2={b2:.3f} se2={se2:.3f} N={n}")

    print("\nNote: Wild cluster bootstrap p-values require boottest package.")
    print("Bootstrap p-values are not replicated (no Python equivalent for wild cluster bootstrap).")
    print("Published bootstrap p-values for reference:")
    for key, pub in published.items():
        print(f"  {key}: bt1={pub['bt1']:.3f}, bt2={pub['bt2']:.3f}")


def table5():
    """Table 5: Citizen Requests and Heterogeneity by Clientelist Relationship."""
    print("\n" + "=" * 70)
    print("TABLE 5: Citizen Requests and Heterogeneity by Clientelist Relationship")
    print("=" * 70)

    stacked = load_stacked()

    # Create interaction variables
    stacked['treat_freq'] = stacked['treatment'] * stacked['frequent_interactor']
    stacked['rain_std_freq'] = stacked['rainfall_std_stacked'] * stacked['frequent_interactor']

    # Published values
    published = {
        1: {
            'treatment': (-0.012, 0.013), 'treat_freq': (-0.092, 0.034),
            'frequent_interactor': (0.116, 0.027),
            'lincom_treat': (-0.104, 0.032),
        },
        2: {
            'rainfall_std_stacked': (-0.024, 0.011), 'rain_std_freq': (-0.007, 0.016),
            'frequent_interactor': (0.071, 0.018),
            'lincom_rain': (-0.031, 0.015),
        },
        3: {
            'treatment': (-0.012, 0.013), 'treat_freq': (-0.092, 0.034),
            'rainfall_std_stacked': (-0.024, 0.010), 'rain_std_freq': (-0.005, 0.016),
            'frequent_interactor': (0.116, 0.027),
            'lincom_treat': (-0.103, 0.032), 'lincom_rain': (-0.029, 0.015),
        },
        4: {
            'treatment': (-0.018, 0.012), 'treat_freq': (-0.053, 0.032),
            'rainfall_std_stacked': (-0.016, 0.009), 'rain_std_freq': (-0.001, 0.015),
            'frequent_interactor': (0.082, 0.025),
            'lincom_treat': (-0.072, 0.031), 'lincom_rain': (-0.018, 0.014),
        },
        5: {
            'treatment': (0.004, 0.010), 'treat_freq': (-0.060, 0.025),
            'rainfall_std_stacked': (-0.010, 0.009), 'rain_std_freq': (-0.016, 0.012),
            'frequent_interactor': (0.070, 0.019),
            'lincom_treat': (-0.056, 0.024), 'lincom_rain': (-0.026, 0.012),
        },
    }

    specs = [
        (1, 'ask_private_stacked', ['frequent_interactor', 'treatment', 'treat_freq']),
        (2, 'ask_private_stacked', ['frequent_interactor', 'rainfall_std_stacked', 'rain_std_freq']),
        (3, 'ask_private_stacked', ['frequent_interactor', 'treatment', 'rainfall_std_stacked', 'treat_freq', 'rain_std_freq']),
        (4, 'ask_nowater_private_stacked', ['frequent_interactor', 'treatment', 'rainfall_std_stacked', 'treat_freq', 'rain_std_freq']),
        (5, 'askrec_private_stacked', ['frequent_interactor', 'treatment', 'rainfall_std_stacked', 'treat_freq', 'rain_std_freq']),
    ]

    for col, depvar, regressors in specs:
        all_X = regressors + ['year2012']
        model, n = ols_with_fe(depvar, all_X, 'mun_id', 'b_clusters', stacked)
        se_adj = get_adj_se(model)

        pub = published[col]
        print(f"\nColumn ({col}): {depvar}, N = {n}")

        for reg in regressors:
            beta = model.params[reg]
            se = se_adj[reg]
            if reg in pub:
                pb, ps = pub[reg]
                match = abs(beta - pb) < 0.0015 and abs(se - ps) < 0.0015
                print(f"  {reg:<35} {beta:8.3f} ({se:.3f})  pub: {pb:8.3f} ({ps:.3f}) {'✓' if match else '✗'}")
            else:
                print(f"  {reg:<35} {beta:8.3f} ({se:.3f})")

        # Linear combinations
        if 'lincom_treat' in pub:
            # treatment + treat_freq
            idx_t = list(model.params.index).index('treatment')
            idx_tf = list(model.params.index).index('treat_freq')
            cov = model.cov_params() * (se_adj.iloc[0]**2 / model.bse.iloc[0]**2) if hasattr(model, '_adj_factor') else model.cov_params()
            beta_lc = model.params.iloc[idx_t] + model.params.iloc[idx_tf]
            var_lc = cov.iloc[idx_t, idx_t] + cov.iloc[idx_tf, idx_tf] + 2 * cov.iloc[idx_t, idx_tf]
            se_lc = np.sqrt(var_lc)
            pb, ps = pub['lincom_treat']
            match = abs(beta_lc - pb) < 0.0015 and abs(se_lc - ps) < 0.0015
            print(f"  {'β1+β2 (treat for clientelist)':<35} {beta_lc:8.3f} ({se_lc:.3f})  pub: {pb:8.3f} ({ps:.3f}) {'✓' if match else '✗'}")

        if 'lincom_rain' in pub:
            # rainfall + rain_std_freq
            idx_r = list(model.params.index).index('rainfall_std_stacked')
            idx_rf = list(model.params.index).index('rain_std_freq')
            cov = model.cov_params() * (se_adj.iloc[0]**2 / model.bse.iloc[0]**2) if hasattr(model, '_adj_factor') else model.cov_params()
            beta_lc = model.params.iloc[idx_r] + model.params.iloc[idx_rf]
            var_lc = cov.iloc[idx_r, idx_r] + cov.iloc[idx_rf, idx_rf] + 2 * cov.iloc[idx_r, idx_rf]
            se_lc = np.sqrt(var_lc)
            pb, ps = pub['lincom_rain']
            match = abs(beta_lc - pb) < 0.0015 and abs(se_lc - ps) < 0.0015
            print(f"  {'β3+β4 (rain for clientelist)':<35} {beta_lc:8.3f} ({se_lc:.3f})  pub: {pb:8.3f} ({ps:.3f}) {'✓' if match else '✗'}")

        # Means
        cols_needed = list(dict.fromkeys([depvar] + all_X + ['mun_id', 'b_clusters', 'treatment', 'frequent_interactor']))
        sample = stacked[cols_needed].dropna()
        mean_treat = sample.loc[sample['treatment'] == 1, depvar].mean()
        mean_ctrl = sample.loc[sample['treatment'] == 0, depvar].mean()
        mean_ctrl_freq = sample.loc[(sample['treatment'] == 0) & (sample['frequent_interactor'] == 1), depvar].mean()
        print(f"  Mean Y (treat): {mean_treat:.3f}, Mean Y (ctrl): {mean_ctrl:.3f}, Mean Y (ctrl, freq): {mean_ctrl_freq:.3f}")


def main():
    table1()
    table2()
    table3()
    table4()
    table5()


if __name__ == '__main__':
    main()
