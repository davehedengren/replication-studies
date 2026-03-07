// =============================================================================
// AERs for AERs — App Configuration & Data
// =============================================================================

const CONFIG = {
  repoUrl: 'https://github.com/davehedengren/replication-studies',
  openicpsrBase: 'https://www.openicpsr.org/openicpsr/project'
};

// =============================================================================
// STUDY DATA — 41 Replication Studies
// =============================================================================

const STUDIES = [
  {
    id: '112318',
    title: 'Does School Choice Lead to Sorting? Evidence from Tiebout Variation',
    authors: 'Urquiola',
    journal: 'American Economic Review',
    year: 2005,
    topic: 'Education',
    verdict: 'replicated',
    mainClaim: 'Greater school choice (more districts per metro area) leads to more racially and educationally homogeneous districts and higher private school enrollment.',
    summaryFindings: 'All sample sizes match exactly. All Table 2\u20134 coefficients replicate within rounding. Minor SE differences in 2 FE specifications attributable to DOF adjustments.',
    tablesReplicated: 'Tables 1\u20135; Figures 1, 3\u20135',
    keyFindings: [
      'All sample sizes match exactly across every specification',
      'All Table 2\u20134 coefficients replicate within rounding tolerance',
      'Placebo permutation test p<0.001 confirms result robustness',
      'Two minor SE discrepancies in FE specs attributable to degrees-of-freedom adjustments'
    ],
    notableIssues: [],
    bugs: [],
    robustnessNotes: 'Regional heterogeneity: effect strongest in the North, insignificant in the South.',
    hasRobustnessConcerns: false,
    images: [],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112319',
    title: 'When Optimal Choices Feel Wrong: A Laboratory Study of Bayesian Updating, Complexity, and Affect',
    authors: 'Charness & Levin',
    journal: 'American Economic Review',
    year: 2005,
    topic: 'Behavioral',
    verdict: 'partial',
    mainClaim: 'When reinforcement learning and Bayesian updating conflict, people frequently violate Bayesian predictions; removing emotional affect substantially reduces errors.',
    summaryFindings: 'Central result confirmed: ~50% switching errors after Left draws vs ~5% after Right draws. Tables 4, 7 match perfectly. Table 8 regression signs and significance match. Small cell-count discrepancies in Tables 1\u20133.',
    tablesReplicated: 'Tables 1\u20138 (Tables 4, 7 exact; Tables 1\u20133 with small discrepancies)',
    keyFindings: [
      'Core result confirmed: ~50% switching errors after Left draws vs ~5% after Right draws',
      'Table 4 (starting errors) and Table 7 (time segments) match perfectly',
      'Table 8 regression coefficients match in sign and significance',
      'Small systematic discrepancies in switching-error counts (1\u20138 per cell) in Tables 1\u20133'
    ],
    notableIssues: [
      'No analysis code provided in replication package \u2014 all code reconstructed from paper descriptions',
      'Small systematic discrepancies possibly from undocumented color-to-variable mapping'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['figure1_starting_errors.png','figure2_right_switching.png','figure3_left_black.png','figure4_left_white.png','figure5_t3_switching.png','figure6_cost_frequency.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112323',
    title: 'Entry, Exit and Investment-Specific Technical Change',
    authors: 'Samaniego',
    journal: 'American Economic Review',
    year: 2008,
    topic: 'Macroeconomics',
    verdict: 'replicated',
    mainClaim: 'The interaction of investment-specific technical change (ISTC) and entry costs suppresses firm turnover, entry, and exit rates more in high-ISTC industries.',
    summaryFindings: 'All 12 full-sample DiD regression coefficients, p-values, sample sizes, and R-squared values replicate exactly. Manufacturing-only subsample coefficients differ from published values, likely due to working paper vs. published AER version discrepancies.',
    tablesReplicated: 'Tables 2, 4\u20135 (full sample exact), Table 11, Figure 1',
    keyFindings: [
      'All 12 full-sample DiD specifications replicate exactly in coefficient, p-value, N, and R-squared',
      'Cross-section correlations between ISTC, turnover, entry, and exit match exactly',
      'Manufacturing-only subsample coefficients differ (working paper vs. AER version)',
      '34 robustness checks all produce negative coefficients; 31/34 significant at 5%'
    ],
    notableIssues: [
      'IV regressions not replicated due to missing legal origin data',
      'Other Mining and Utilities have identical values in industry file, possibly duplicated'
    ],
    bugs: [],
    robustnessNotes: 'The main finding is highly robust across 34 specifications. Placebo permutation test p=0.001.',
    hasRobustnessConcerns: false,
    images: ['figure1_istc_turnover.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112783',
    title: 'Entrepreneurial Innovation: Killer Apps in the iPhone Ecosystem',
    authors: 'Yin, Davis & Muzyrya',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Behavioral',
    verdict: 'replicated',
    mainClaim: 'Experience and no updating increase the likelihood of a game becoming a killer app, while more updates increase the likelihood for non-game killer apps, suggesting opposing innovation processes.',
    summaryFindings: 'All probit coefficients, standard errors, pseudo R-squared values, and sample sizes replicate exactly across Table 2. Marginal effects show small differences (~10\u201315%) due to Stata margeff vs. statsmodels methodology.',
    tablesReplicated: 'Table 1 (exact), Table 2 probit coefficients (exact)',
    keyFindings: [
      'All probit coefficients, standard errors, sample sizes, and pseudo R-squared replicate exactly',
      'App Order effect for games is the most robust finding, surviving all 9 specification checks',
      'Number of Versions effect for non-games loses significance in 3 of 9 robustness checks',
      '1,085 exact duplicate rows (3.0% of data) are present but do not affect results'
    ],
    notableIssues: [
      '1,085 exact duplicate rows (3.0%) in the data, unexplained',
      '47.6% of observations have zero comments AND zero score'
    ],
    bugs: [],
    robustnessNotes: 'The Number of Versions effect for non-games loses significance when dropping outliers or restricting to early/late cohorts only. The App Order effect for games is robust across all checks.',
    hasRobustnessConcerns: true,
    images: [],
    codeFiles: ['01_clean.py','02_tables.py','03_data_audit.py','04_robustness.py','utils.py']
  },
  {
    id: '112786',
    title: 'Rainfall Forecasts, Weather and Wages over the Agricultural Production Cycle',
    authors: 'Rosenzweig & Udry',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Development',
    verdict: 'replicated',
    mainClaim: 'IMD long-range monsoon rainfall forecasts reduce planting-stage migration and wages in India, and a good forecast exacerbates the negative effect of bad realized rainfall on harvest wages.',
    summaryFindings: 'All regression coefficients in Tables 2 and 3 replicate exactly within rounding tolerance. Figure 1 cannot be precisely replicated due to a missing year variable in the data.',
    tablesReplicated: 'Tables 2\u20133 (exact)',
    keyFindings: [
      'All coefficients and t-statistics in Tables 2 and 3 match exactly',
      'Planting-stage wage effects are highly robust across all checks',
      'Harvest-stage forecast x bad rainfall interaction drops below 5% significance with robust SEs',
      'Migration-forecast result is driven by a single village'
    ],
    notableIssues: [
      'Paper uses conventional (non-robust) standard errors; HC1 SEs make several harvest results insignificant',
      'Migration result driven by one village out of six'
    ],
    bugs: [],
    robustnessNotes: 'Planting-stage results are robust. Harvest-stage results are fragile: forecast x bad rainfall loses significance with HC1 SEs (t=1.76) and distance restriction (t=1.27). Migration is driven by one village.',
    hasRobustnessConcerns: true,
    images: ['figure1_wage_ratio.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112788',
    title: 'Bundling Health Insurance and Microfinance in India: There Cannot be Adverse Selection if There Is No Demand',
    authors: 'Banerjee, Duflo & Hornbeck',
    journal: 'American Economic Review',
    year: 2014,
    topic: 'Development',
    verdict: 'replicated',
    mainClaim: 'Mandatory health insurance bundled with microfinance reduced loan takeup by 23% with no evidence of adverse selection.',
    summaryFindings: 'Every coefficient, standard error, and sample size matches exactly. All 17 interaction coefficients for adverse selection tests match. Joint F-test insignificant (p=0.44). Robust across all 12 robustness checks.',
    tablesReplicated: 'Tables 1\u20133; all in-text statistics',
    keyFindings: [
      'Treatment reduced loan renewal by 16pp (\u03b2=-0.161, SE=0.024) \u2014 exact match',
      'All 17 adverse selection interaction coefficients match exactly',
      'Joint F-test for adverse selection insignificant (p=0.44)',
      'Exemplary replication: 100% exact match across all specifications'
    ],
    notableIssues: [],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: [],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112792',
    title: 'Disability Insurance and Health Insurance Reform: Evidence from Massachusetts',
    authors: 'Maestas, Mullen & Strand',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Public Economics',
    verdict: 'replicated',
    mainClaim: 'Massachusetts\u2019 2006 health reform caused a modest, temporary increase in disability applications, driven by SSDI-only applications.',
    summaryFindings: 'All Tables 1, 2, and A-1 replicate exactly. MA reform led to ~3% increase in SSDI applications in FY2008. County-level heterogeneity by insurance coverage confirmed.',
    tablesReplicated: 'Tables 1 (Panels A\u2013D), 2, A-1',
    keyFindings: [
      'All coefficients and standard errors match exactly across all specifications',
      'MA reform led to ~3% increase in SSDI applications concentrated in FY2008',
      'County-level heterogeneity by prior insurance coverage confirmed'
    ],
    notableIssues: [
      'Permutation test p=0.44 (low power with only 9 comparison states)',
      'Two significant pre-treatment event study coefficients weaken parallel trends assumption'
    ],
    bugs: [],
    robustnessNotes: 'Pre-treatment event study coefficients and low-power permutation test raise questions about causal identification.',
    hasRobustnessConcerns: true,
    images: ['fig1_trends.png','fig2_coef_plot.png','fig3_heterogeneity.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112798',
    title: 'Are Female Supervisors More Female-Friendly?',
    authors: 'Bednar & Gicheva',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Labor',
    verdict: 'replicated',
    mainClaim: 'Supervisor gender (athletic director) is not significantly related to the propensity to hire and retain female coaches in NCAA Division I athletics.',
    summaryFindings: 'All key null results replicate \u2014 female AD gender is not significantly related to female-friendliness across all specifications. R-squared essentially zero. Null finding robust across all 10 robustness checks.',
    tablesReplicated: 'Tables 1\u20133; Figure 1',
    keyFindings: [
      'Null result confirmed: female AD gender has no significant effect across all specifications',
      'R-squared essentially zero across all models',
      'Null finding robust across 10 additional robustness checks'
    ],
    notableIssues: [
      'Only 42 female ADs (9.7%) in sample limits statistical power'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['figure1_histogram.png'],
    codeFiles: ['01_tables.py','02_figure1.py','03_data_audit.py','04_robustness.py','utils.py']
  },
  {
    id: '112799',
    title: 'Race and Marriage in the Labor Market: A Discrimination Correspondence Study in a Developing Country',
    authors: 'Arceo-Gomez & Campos-Vazquez',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Labor',
    verdict: 'replicated',
    mainClaim: 'There is significant gender, racial, and marital status discrimination in Mexico City\u2019s labor market, with women favored overall but indigenous-phenotype women penalized.',
    summaryFindings: 'All core table coefficients match within 0.001. Women receive ~4.3pp more callbacks; European-phenotype women receive ~3.3pp more than indigenous-phenotype women; marriage penalty for women (~2.8pp) but not men.',
    tablesReplicated: 'Tables 4\u20137, Appendix B1, C1',
    keyFindings: [
      'Women receive ~4.3pp more callbacks than men \u2014 exact match',
      'European-phenotype women receive ~3.3pp more callbacks than indigenous-phenotype women',
      'Marriage penalty exists for women (~2.8pp) but not men',
      'All core coefficients match within 0.001'
    ],
    notableIssues: [
      'Block randomization creates correlations between treatment and control variables (placebo test fails)',
      'Marriage penalty driven entirely by private university graduates'
    ],
    bugs: [],
    robustnessNotes: 'Failed placebo test suggests treatment-control correlations from block randomization may bias estimates.',
    hasRobustnessConcerns: true,
    images: [],
    codeFiles: ['01_clean.py','02_tables.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112801',
    title: 'The Declining Fortunes of the Young Since 2000',
    authors: 'Beaudry, Green & Sand',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Labor',
    verdict: 'replicated',
    mainClaim: 'Successive cohorts of college-educated workers entering after 2000 experienced declining shares of cognitive employment, reversing the 1990s trend.',
    summaryFindings: 'All three figures replicate. Cognitive employment share for BA entry cohorts rose from ~0.55 (1990) to ~0.62 (2000), then reversed to ~0.55 (2010). Pattern confirmed.',
    tablesReplicated: 'Figures 1 (A/B), 2, 3',
    keyFindings: [
      'Cognitive employment share patterns replicate exactly across all cohorts',
      'Clear reversal point at 2000 entry cohort confirmed',
      'Decline driven primarily by male workers; females show virtually no difference'
    ],
    notableIssues: [
      'Trend break regression only marginally significant (p=0.07)',
      'Post-college workers show no decline, suggesting the pattern is education-specific'
    ],
    bugs: [],
    robustnessNotes: 'Decline is entirely a male phenomenon \u2014 female workers show no decline in cognitive employment shares.',
    hasRobustnessConcerns: true,
    images: ['fig1a_cognitive_emp_smoothed_BA.png','fig1a_cognitive_emp_raw_BA.png','fig1b_wages_smoothed_BA.png','fig1b_wages_raw_BA.png','fig2_employment_shares_5yr.png','fig3a_cognitive_emp_smoothed_Post-College.png','fig3a_cognitive_emp_raw_Post-College.png','fig3b_wages_smoothed_Post-College.png','fig3b_wages_raw_Post-College.png'],
    codeFiles: ['01_clean.py','02_figures.py','03_data_audit.py','04_robustness.py','utils.py']
  },
  {
    id: '112802',
    title: 'Trends in Earnings Differentials Across College Majors and the Changing Task Composition of Jobs',
    authors: 'Altonji, Kahn & Speer',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Labor',
    verdict: 'replicated',
    mainClaim: 'Earnings differentials across college majors widened significantly from 1993\u20132003, with changing returns to task content accounting for about two-thirds of the increase.',
    summaryFindings: 'All Table 1 and Table 2 values match to within 0.001. Earnings differentials widened 24% (1993\u20132003), task measures explain ~70%.',
    tablesReplicated: 'Tables 1, 2',
    keyFindings: [
      'All coefficients match to within 0.001 across both tables',
      'Earnings differentials widened 24% during 1993\u20132003',
      'Task measures explain approximately 70% of the widening',
      'Permutation test: 1993\u20132003 widening significant (p=0.00) but 1993\u20132011 is not (p=0.17)'
    ],
    notableIssues: [
      'NSCG observations receive ~20x weight amplification relative to ACS observations'
    ],
    bugs: [],
    robustnessNotes: '1993\u20132011 trend driven entirely by older workers and fails permutation test, suggesting the widening may not have persisted.',
    hasRobustnessConcerns: true,
    images: [],
    codeFiles: ['01_clean.py','02_tables.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112803',
    title: 'Return of the Solow Paradox? IT, Productivity, and Employment in U.S. Manufacturing',
    authors: 'Acemoglu, Autor, Dorn, Hanson & Price',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Labor',
    verdict: 'replicated',
    mainClaim: 'IT-intensive manufacturing industries show no net productivity gains; apparent labor productivity gains are driven by employment declines, not output increases.',
    summaryFindings: 'All 8 Table A4 coefficients match exactly. IT-intensive industries show no net productivity gain by 2009. Placebo test confirms insignificance (p=0.78).',
    tablesReplicated: 'Table A4 (8 specifications); Figures 1\u20133',
    keyFindings: [
      'All 8 Table A4 coefficients match exactly',
      'IT-intensive industries show no net productivity gain by 2009',
      'Placebo test confirms insignificance of the relationship (p=0.78)'
    ],
    notableIssues: [
      'Baseline coefficient is statistically insignificant',
      'Substantial cross-sector heterogeneity in IT effects'
    ],
    bugs: [],
    robustnessNotes: 'Results are sensitive to IT measure vintage \u2014 using ci7782 instead of ci8292 produces a positive coefficient, reversing the sign.',
    hasRobustnessConcerns: true,
    images: ['Figure_1A.png','Figure_1B.png','Figure_2A.png','Figure_3A.png','robustness_summary.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112805',
    title: 'Using School Choice Lotteries to Test Measures of School Effectiveness',
    authors: 'Deming',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Education',
    verdict: 'replicated',
    mainClaim: 'School value-added models with prior test score controls and multiple years of data are accurate, unbiased predictors of causal effects on student achievement.',
    summaryFindings: 'All 6 Table 1 coefficients match within 0.01\u20130.07. VAMs with gains specifications and 2+ years of data produce coefficients indistinguishable from 1. Placebo test: 0.008 (passes).',
    tablesReplicated: 'Tables 1, A1, A2, A3',
    keyFindings: [
      'VAM gains specification with 2+ years: coefficient 0.898 (published 0.966) \u2014 close match',
      'Model 1 (levels, no controls) near zero as expected',
      'Placebo test coefficient: 0.008 (passes)',
      'Small coefficient differences documented by author as due to de-identified public-use data'
    ],
    notableIssues: [],
    bugs: [],
    robustnessNotes: 'All 19 robustness checks confirm the main findings.',
    hasRobustnessConcerns: false,
    images: [],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112812',
    title: 'Benefits of Neuroeconomic Modeling: New Policy Interventions and Predictors of Preference',
    authors: 'Krajbich, Oud & Fehr',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Behavioral',
    verdict: 'replicated',
    mainClaim: 'Decision-makers misallocate time to low-stakes choices; time-constraint interventions improve choice quality, and response times predict indifference points.',
    summaryFindings: 'All key statistics match exactly: RT by stakes (1.65/1.11s), missed trials (~44), money left on table ($20.10 CHF), and 16/18 probabilistic UG subjects.',
    tablesReplicated: 'All in-text statistics; Figures 3A, 3B, 4',
    keyFindings: [
      'Response time by stakes matches exactly: 1.65s (low) vs 1.11s (high)',
      'Money left on table: $20.10 CHF \u2014 exact match',
      '16/18 probabilistic UG subjects confirmed exactly',
      'All key in-text statistics replicate perfectly'
    ],
    notableIssues: [
      'UG study underpowered (18 subjects, 3\u20135 trials per offer level)'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: [],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112818',
    title: 'Why Has U.S. Policy Uncertainty Risen Since 1960?',
    authors: 'Baker, Bloom, Canes-Wrone, Davis & Rodden',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Political Economy',
    verdict: 'replicated',
    mainClaim: 'U.S. policy uncertainty, government activity, and political polarization all show co-trending upward secular patterns since 1960.',
    summaryFindings: 'Both figures replicate exactly. High correlations (0.79\u20130.92) between EPU, regulation, and polarization confirmed.',
    tablesReplicated: 'Figures 1, 2',
    keyFindings: [
      'Both published figures replicate exactly',
      'Correlations of 0.79\u20130.92 between EPU, regulation, and polarization confirmed',
      'Detrended correlations drop to 0.26\u20130.29',
      'First-differenced correlations are near zero'
    ],
    notableIssues: [
      'Within-decade correlations often negative'
    ],
    bugs: [],
    robustnessNotes: 'Co-movement is entirely driven by shared secular trends. Detrended and first-differenced correlations are near zero, suggesting no causal relationship beyond common upward trends.',
    hasRobustnessConcerns: true,
    images: ['figure1.png','figure2.png'],
    codeFiles: ['01_clean.py','02_figures.py','03_data_audit.py','04_robustness.py','utils.py']
  },
  {
    id: '112819',
    title: 'The Economic Cost of Global Fuel Subsidies',
    authors: 'Davis',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Energy',
    verdict: 'replicated',
    mainClaim: 'Global fuel subsidies cost ~$110 billion annually, generating $44 billion in deadweight loss and $92 billion when externalities are included.',
    summaryFindings: 'All Table A1 country values match exactly. Total subsidies=$110B, DWL=$44B, SMC DWL=$92B \u2014 all confirmed.',
    tablesReplicated: 'Table A1 (both panels); all in-text statistics',
    keyFindings: [
      'Total subsidies = $110B, DWL = $44B, SMC DWL = $92B \u2014 all match exactly',
      'All Table A1 country-level values match',
      'Saudi Arabia + Venezuela account for 50% of total deadweight loss'
    ],
    notableIssues: [],
    bugs: [],
    robustnessNotes: 'Results highly concentrated: 2 countries account for 50% of DWL. Sensitive to demand elasticity assumptions and spot price benchmark choice.',
    hasRobustnessConcerns: true,
    images: ['figure1_gas_consumption.png','figure2_subsidies.png','figure3_dwl.png','figureA1_diesel_consumption.png','figureA3_subsidies_pc.png','figureA4_dwl_pc.png','figureA5_dwl_smc.png','figure_gas_prices_bar.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112820',
    title: 'Why Are Power Plants in India Less Efficient than Power Plants in the United States?',
    authors: 'Chan, Cropper & Malik',
    journal: 'AEA Papers and Proceedings',
    year: 2014,
    topic: 'Energy',
    verdict: 'partial',
    mainClaim: 'State-owned Indian coal power plants are approximately 9.4% less thermally efficient than comparable US plants, with the gap narrowing over time.',
    summaryFindings: 'India treatment effect of ~9.4% efficiency gap matches (0.0945 vs published ~0.094). Year-by-year declining trend matches. Matching estimates 10\u201315% lower than published.',
    tablesReplicated: 'Table A1, Figure 1; Tables A2\u2013A3 (directionally consistent)',
    keyFindings: [
      'India efficiency gap coefficient: 0.0945 vs published ~0.094 \u2014 near-exact match',
      'Year-by-year declining trend in efficiency gap matches Figure 1',
      'Matching estimates are 10\u201315% lower than published values'
    ],
    notableIssues: [
      'Code bug: nnmatch references nonexistent bias variable',
      'Code bug: distribution.do references wrong variable names',
      'India OPHR reporting rate declines from 84% to 54%, raising selection concerns',
      'Age polynomial inconsistency in regression code'
    ],
    bugs: [
      'nnmatch command references a bias variable that does not exist in the dataset',
      'distribution.do references incorrect variable names'
    ],
    robustnessNotes: 'Multiple code bugs and declining data reporting rates raise questions about the precision of matching estimates.',
    hasRobustnessConcerns: true,
    images: [],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112874',
    title: 'Too-Systemic-To-Fail: What Option Markets Imply About Sector-Wide Government Guarantees',
    authors: 'Kelly, Lustig & Van Nieuwerburgh',
    journal: 'American Economic Review',
    year: 2016,
    topic: 'Macroeconomics',
    verdict: 'replicated',
    mainClaim: 'The financial sector basket-index put spread rises dramatically during the 2007\u20132009 crisis, consistent with investors pricing a collective bailout guarantee for the financial sector.',
    summaryFindings: 'Python code matches MATLAB pre-computed values with 0.000% error on all basket-index spread calculations. All core empirical results replicate exactly. Minor level differences from delta=25 in replication vs. delta=20 in published paper.',
    tablesReplicated: 'Tables 1\u20133, 7\u20139A; Figures 1\u20138',
    keyFindings: [
      'Basket-index put spread for financials rises 4-fold from pre-crisis to crisis \u2014 exact match',
      'No comparable increase for non-financial sectors or call options \u2014 bailout-specific pattern confirmed',
      'BS model captures only 41% of crisis increase; Merton-Jump bailout model captures 88%',
      'All 11 robustness checks confirm core findings'
    ],
    notableIssues: [
      'Replication code uses delta=25 while published paper reports delta=20, causing systematic level differences'
    ],
    bugs: [],
    robustnessNotes: 'All core findings survive every robustness check: alternative deltas, crisis dates, winsorization, dropping crisis peak, leave-one-sector-out, and placebo call spreads.',
    hasRobustnessConcerns: false,
    images: ['figure1_basket_index_cpdi.png','figure2_put_call_spread.png','figure3_bs_correlation.png','figure4_correlations.png','figure5_iv_skew.png','figure6_mj_fits_by_sector.png','figure7_announcements.png','figure8_spread_by_sector.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112875',
    title: 'Institutional Corruption and Election Fraud: Evidence from a Field Experiment in Afghanistan',
    authors: 'Callen & Long',
    journal: 'American Economic Review',
    year: 2015,
    topic: 'Political Economy',
    verdict: 'replicated',
    mainClaim: 'Announcing photo quick count election monitoring reduced fraud in Afghanistan\u2019s 2010 parliamentary elections: votes for connected candidates fell by ~25%, and form theft fell by ~60%.',
    summaryFindings: 'All main results replicate exactly across all 9 main-text tables. Every coefficient, standard error, and sample size matches published values.',
    tablesReplicated: 'Tables 1\u20139; Figures A1\u2013A3',
    keyFindings: [
      'All published results across Tables 1\u20139 replicate exactly \u2014 exceptionally clean replication',
      'Form theft result (60% reduction) is very robust across all specifications',
      'Treatment effect on connected candidates\u2019 votes becomes insignificant when winsorized at 1% or 5%',
      'Spatial externalities confirm treatment diffusion to nearby polling centers'
    ],
    notableIssues: [
      'Treatment effect on connected candidates\u2019 votes is sensitive to winsorization',
      'Only 57 candidates have political connection data; 19 are most connected'
    ],
    bugs: [],
    robustnessNotes: 'Form theft result is very robust. The vote result for connected candidates is fragile: winsorizing at 1% makes it insignificant (t=-1.42), and at 5% even more so (t=-0.88). Permutation test is borderline (p=0.048).',
    hasRobustnessConcerns: true,
    images: ['figureA1.png','figureA2.png','figureA3.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '112876',
    title: 'The Anatomy of a Credit Crisis: The Boom and Bust in Farm Land Prices in the United States in the 1920s',
    authors: 'Rajan & Ramcharan',
    journal: 'American Economic Review',
    year: 2015,
    topic: 'Macroeconomics',
    verdict: 'replicated',
    mainClaim: 'The expansion of the U.S. banking system in the early 1920s inflated farmland prices, and the subsequent credit contraction caused a persistent collapse in land values lasting through at least 1972.',
    summaryFindings: 'All regression coefficients across Tables 5A, 5B, 6, 9\u201312, and Figure 5 replicate exactly within rounding tolerance. The bank coefficient drops 61% when controls are added, and the panel specification is fragile to dropping census years.',
    tablesReplicated: 'Tables 5A\u20136, 9\u201312, Figure 5 (all exact)',
    keyFindings: [
      'All regression coefficients and t-statistics match exactly across all replicated tables',
      'Cross-sectional banks-to-land-price relationship is highly robust across leave-one-state-out',
      'Panel specification becomes insignificant when dropping year 1900 (t from 2.48 to 0.58)',
      'Long-run reversal and bank suspension persistence effects are strongly supported'
    ],
    notableIssues: [
      'Bank coefficient drops 61% (0.60 to 0.23) when controls added, suggesting substantial confounding',
      'Panel estimate depends on having all three census years (1900, 1910, 1920)',
      'IV/OLS ratio of 1.39 could indicate measurement error or weak instrument concerns'
    ],
    bugs: [],
    robustnessNotes: 'Cross-sectional results (Table 5A) are highly robust. Panel specification is fragile to dropping census years. State-clustered SEs are 1.8x HC1 SEs.',
    hasRobustnessConcerns: true,
    images: [],
    codeFiles: ['02_tables.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '164661',
    title: 'When Choices Are Mistakes',
    authors: 'Nielsen & Rehbeck',
    journal: 'American Economic Review',
    year: 2022,
    topic: 'Behavioral',
    verdict: 'replicated',
    mainClaim: 'When shown violations of canonical choice axioms, decision-makers treat them as mistakes and revise choices to be consistent with the axioms.',
    summaryFindings: 'Tables I, II, IV match exactly. ~85% select canonical axioms; 79% of revisions favor the axiom. FOSD accounts for 41% of violations but has lowest revision rate (51%).',
    tablesReplicated: 'Tables I, II, IV; Figures IV\u2013VII',
    keyFindings: [
      'All main table values match exactly',
      '~85% of subjects select canonical axioms when presented',
      '79% of revisions favor the axiom over the violation',
      'FOSD dominance \u2014 dropping it raises aggregate favor rate from 79% to 90%'
    ],
    notableIssues: [
      'Minor p-value differences in Fisher exact and signed-rank tests'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: [],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '173341',
    title: 'Vulnerability and Clientelism',
    authors: 'Bobonis, Gertler, Gonzalez-Navarro & Nichter',
    journal: 'American Economic Review',
    year: 2022,
    topic: 'Political Economy',
    verdict: 'partial',
    mainClaim: 'Water cisterns reduce household vulnerability and decrease clientelistic exchanges with politicians, concentrated among those with pre-existing clientelist relationships.',
    summaryFindings: 'Table 2 (all 9 specifications) matches exactly. Table 4 coefficients within 0.002. Data version discrepancy for one postcoded variable affects Tables 3 and 5.',
    tablesReplicated: 'Tables 1\u20135 (Tables 3, 5 with data version discrepancy)',
    keyFindings: [
      'Table 2 (9 specifications) matches exactly',
      'Table 4 coefficients within 0.002 of published values',
      'Data version discrepancy: shipped data has different values for c_ask_private_pol_postcoded (mean=0.213 vs published 0.195)'
    ],
    notableIssues: [
      'Shipped data has different values for one postcoded variable versus published .tex outputs',
      'Packaging issue, not a code bug'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['figure_a1.png','figure_a2.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '179162',
    title: 'Detecting Mother-Father Differences in Spending on Children: A New Approach Using Willingness-to-Pay Elicitation',
    authors: 'Dizon-Ross & Jayachandran',
    journal: 'AER: Insights',
    year: 2023,
    topic: 'Development',
    verdict: 'replicated',
    mainClaim: 'Fathers spend significantly less on daughters than sons, while mothers do not discriminate by child gender.',
    summaryFindings: 'All main coefficients match to 3rd decimal place (girl: -0.102, momXgirl: +0.132). Sample size matches within 1 observation (6,672 vs 6,673).',
    tablesReplicated: 'Table 1 (6 columns); Figures 2, 3a, 3b, 4a, 4b',
    keyFindings: [
      'Girl coefficient: -0.102 \u2014 exact match to 3rd decimal',
      'MomXGirl interaction: +0.132 \u2014 exact match',
      'Sample size: 6,672 vs published 6,673 (within 1 observation)',
      'momXgirl interaction disappears for incentivized goods and in follow-up wave'
    ],
    notableIssues: [
      'Coding quirk assigns daughter=1 for children with missing gender'
    ],
    bugs: [],
    robustnessNotes: 'The key momXgirl interaction is driven entirely by non-incentivized goods. It disappears for incentivized goods and in the follow-up wave, raising questions about the robustness of the gender-differential finding.',
    hasRobustnessConcerns: true,
    images: ['figure2.png','figure3a.png','figure3b.png','figure4a.png','figure4b.png'],
    codeFiles: ['01_clean.py','02_table1.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '192297',
    title: 'Big Loans to Small Businesses: Predicting Winners and Losers in an Entrepreneurial Lending Experiment',
    authors: 'Bryan, Karlan & Osman',
    journal: 'American Economic Review',
    year: 2024,
    topic: 'Development',
    verdict: 'replicated',
    mainClaim: 'Larger microfinance loans have no significant average effect on profits, but ML identifies clear winners and losers among borrowers.',
    summaryFindings: 'Tables 1\u20134 replicate closely. Core finding confirmed \u2014 positive but insignificant ATE on profits with strong ML-predicted heterogeneity. Critical coding bug discovered: profit variable double-counts primary business profits.',
    tablesReplicated: 'Tables 1\u20134, Figure 1',
    keyFindings: [
      'Core finding confirmed: insignificant ATE with strong ML-predicted heterogeneity',
      'ML group assignments replicate: G4 winners (+18,333 EGP) and G1 losers (-13,642 EGP)',
      'Critical coding bug: e_month_profit_all double-counts primary business profits via Stata wildcard rowtotal',
      'Bug inflates ATE by ~40% (1,294 vs 788 EGP corrected) but does not change ML group assignments'
    ],
    notableIssues: [
      'Stata wildcard rowtotal(e_month_profit_*) inadvertently includes the primary business profit twice',
      'Affects Table 4 profits row, Table 5, and Appendix A10'
    ],
    bugs: [
      'e_month_profit_all double-counts primary business profits due to Stata wildcard rowtotal matching both the individual variable and the total. Inflates ATE by approximately 40%.'
    ],
    robustnessNotes: 'The coding bug inflates the average treatment effect by ~40%, though qualitative conclusions and ML group assignments are unaffected.',
    hasRobustnessConcerns: true,
    images: ['figure1_outstanding_debt.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figure.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '208367',
    title: 'Shock Sizes and the Marginal Propensity to Consume',
    authors: 'Andreolli & Surico',
    journal: 'American Economic Review',
    year: 2025,
    topic: 'Macroeconomics',
    verdict: 'partial',
    mainClaim: 'MPC from small income shocks exceeds MPC from large shocks, with the difference concentrated among cash-poor households.',
    summaryFindings: 'Sample size matches exactly (N=4,524). MPC means match (small: 0.473, large: 0.439, diff: 0.034). Tobit decile patterns match. Figures 5\u20136 and Table I.1 require MATLAB output files not included in package.',
    tablesReplicated: 'Tables 1\u20132, D.6, 4; Figures 1\u20134',
    keyFindings: [
      'Sample size: N=4,524 \u2014 exact match',
      'MPC means match: small shocks 0.473, large shocks 0.439, difference 0.034',
      'Tobit decile patterns match published results',
      '3 MATLAB output files missing from replication package prevent full replication'
    ],
    notableIssues: [
      'Three MATLAB output files missing from package (needed for Figures 5\u20136, Table I.1)',
      'No code bugs found in available materials'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['figure_1_mpc_histograms.png','figure_2_mpc_percentiles.png','figure_3_eatout_cash.png','figure_4_binscatter_mpc_eatout.png','figure_mpc_deciles.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '208722',
    title: 'Terrorism and Voting: The Rise of Right-Wing Populism in Germany',
    authors: 'Sabet, Liebald & Friebel',
    journal: 'AEJ: Economic Policy',
    year: 2025,
    topic: 'Political Economy',
    verdict: 'replicated',
    mainClaim: 'Successful terror attacks increase right-wing populist (AfD) vote share by ~2 percentage points in German federal elections.',
    summaryFindings: 'Baseline DiD estimate of ~2pp increase in AfD vote share confirmed (N=797, 124 municipalities). Failed attacks show no effect (placebo passes). Right-wing attacks show stronger effect (+3.7pp).',
    tablesReplicated: 'Tables 1A, 1B, 2, 3; Figure 3 (event study)',
    keyFindings: [
      'Baseline DiD: ~2pp increase in AfD vote share \u2014 confirmed',
      'Failed attacks show no effect (placebo passes)',
      'Right-wing attacks show stronger effect (+3.7pp)',
      'Event study dynamics consistent with parallel trends pre-treatment'
    ],
    notableIssues: [
      'Small control group (11 municipalities with failed attacks)',
      'SOEP tables (4\u20135) require confidential data'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['figure_3_event_study.png','figure_2_placebo_events.png','figure_afd_distribution.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '212821',
    title: 'Underestimating Learning by Doing',
    authors: 'Horn & Loewenstein',
    journal: 'AEJ: Microeconomics',
    year: 2025,
    topic: 'Behavioral',
    verdict: 'replicated',
    mainClaim: 'People systematically underpredict their future task performance improvement through practice; outside predictors do not show this bias.',
    summaryFindings: 'All Table 1 Panel A values match to 2 decimal places. All Panel C slopes match. Table 2 performer/predictor comparisons match exactly. Data is 100% complete with zero missing values.',
    tablesReplicated: 'Tables 1\u20132, A1\u2013A5; Figures 1\u20133, A3',
    keyFindings: [
      'All 9 key values match perfectly to 2 decimal places',
      'Panel C slopes match exactly across all specifications',
      'Table 2 performer vs predictor comparisons: exact match',
      'Exceptionally clean replication package \u2014 zero missing values in entire dataset'
    ],
    notableIssues: [],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['Figure1.png','Figure2.png','Figure3.png','FigureA3.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '219907',
    title: 'Labor Market Power, Self-Employment, and Development',
    authors: 'Amodio, Medina & Morlacco',
    journal: 'American Economic Review',
    year: 2025,
    topic: 'Labor',
    verdict: 'replicated',
    mainClaim: 'Higher employer concentration increases self-employment rates and reduces wages in Peruvian manufacturing, consistent with monopsony pushing workers into self-employment.',
    summaryFindings: 'OLS and IV estimates match in sign and significance. IV inverse labor supply elasticity: 0.455 (SE=0.133, F=17.87). Mean HHI=0.65 with 38.7% single-firm markets.',
    tablesReplicated: 'Tables 1, 2, A2; Figures 1, 2, B3',
    keyFindings: [
      'IV inverse labor supply elasticity: 0.455 (SE=0.133, F=17.87) \u2014 matches',
      'Mean HHI = 0.65 with 38.7% single-firm markets confirmed',
      'OLS and IV signs and significance match across all specifications'
    ],
    notableIssues: [
      'Only 2.5% of workers (7,637/307,566) have HHI data after merging',
      'Minor coefficient differences from AbsorbingLS vs reghdfe'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['figure_1_transitions.png','figure_2_binscatter.png','figure_b3_hhi_correlations.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '221423',
    title: 'Income Inequality in the Nordic Countries: Myths, Facts, and Lessons',
    authors: 'Mogstad, Salvanes & Torsvik',
    journal: 'American Economic Review',
    year: 2025,
    topic: 'Public Economics',
    verdict: 'partial',
    mainClaim: 'Nordic countries achieve comparable GDP per capita to the US with far less income inequality, enabled by high redistribution, union density, and employment rates.',
    summaryFindings: '60/60 values in Table 1 match exactly. Table 3 Gini coefficients match for all 3 replicable columns. Nordic Gini disposable=0.27 vs US=0.39.',
    tablesReplicated: 'Table 1 (15/16 rows), Table 3 (3/5 columns), Figures 2\u20133',
    keyFindings: [
      '60 of 60 values in Table 1 match exactly',
      'Table 3 Gini coefficients match for all replicable columns',
      'Nordic disposable income Gini = 0.27 vs US = 0.39 \u2014 confirmed'
    ],
    notableIssues: [
      '11 of 14 tables require PIAAC microdata (OECD registration required)',
      'Code uses 2021 public sector employment data despite paper referencing 2019'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['figure2.png','figure3a.png','figure3b.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '225841',
    title: 'Do Credit Conditions Move House Prices?',
    authors: 'Greenwald & Guren',
    journal: 'American Economic Review',
    year: 2025,
    topic: 'Macroeconomics',
    verdict: 'partial',
    mainClaim: 'Credit supply shocks causally affect house prices and homeownership rates, with effects building over several years.',
    summaryFindings: 'LP-IRF methodology successfully replicated using non-confidential pseudodata. Homeownership rate effect is positive and significant at short horizons, qualitatively matching the paper. Exact coefficient comparison not possible because published results use confidential microdata.',
    tablesReplicated: 'Figure 3 equivalent LP-IRFs (qualitative match only due to pseudodata)',
    keyFindings: [
      'LP-IRF methodology correctly implemented with FWL demeaning, cluster-robust SEs, and population weights',
      'Homeownership rate effect is positive and significant at h=0 and h=1, qualitatively matching the paper',
      'Exact coefficient comparison impossible because published results use confidential GG microdata',
      'Adding employment controls reduces the h=0 homeownership coefficient and it loses significance'
    ],
    notableIssues: [
      'Non-confidential pseudodata precludes exact coefficient comparison',
      'HVS-based samples have only 41\u201362 CBSAs, limiting statistical power'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: [],
    codeFiles: ['02_tables.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '226781',
    title: 'Trade, Value Added, and Productivity Linkages: A Quantitative Analysis',
    authors: 'de Soyres & Gaillard',
    journal: 'American Economic Review',
    year: 2025,
    topic: 'International',
    verdict: 'replicated',
    mainClaim: 'Intermediate goods trade, not final goods trade, drives business cycle comovement across countries.',
    summaryFindings: 'All 4 empirical tables match to 4 decimal places. Intermediate goods trade intensity significantly predicts GDP comovement (0.066, p<0.05); final goods trade does not.',
    tablesReplicated: 'Tables 4, 6, 9, 10',
    keyFindings: [
      'All coefficients match to 4 decimal places across all specifications',
      'Intermediate goods trade coefficient: 0.066 (p<0.05) \u2014 significant',
      'Final goods trade coefficient: not significant \u2014 confirmed',
      'Cleanest replication package encountered across all 32 studies'
    ],
    notableIssues: [
      'Model-based tables (1\u20133, 5, 7, 8, 11) require MATLAB/Dynare'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['figure_trade_comovement.png','figure_trade_trends.png','figure_gdp_corr_dist.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '227802',
    title: 'Filling the Gaps: Childcare Laws for Women\'s Economic Empowerment',
    authors: 'Anukriti, Dinarte-Diaz, Montoya-Aguirre & Sakhonchik',
    journal: 'AEA Papers and Proceedings',
    year: 2025,
    topic: 'Development',
    verdict: 'partial',
    mainClaim: 'Childcare laws increase female labor force participation by ~1.4 pp, with affordability provisions showing strongest effects.',
    summaryFindings: 'Sample sizes and control means match exactly (N=4,960, control mean=64.63). TWFE estimates used instead of synthdid are 15\u201320% larger, as expected. Event study dynamics match.',
    tablesReplicated: 'Tables 1, 2, A1; Figure 2 (event study)',
    keyFindings: [
      'Sample size N=4,960 and control mean 64.63 \u2014 exact match',
      'TWFE estimates 15\u201320% larger than published synthdid (expected with different estimator)',
      'Event study dynamics match published patterns'
    ],
    notableIssues: [
      'synthdid R package used in original; TWFE used as approximation',
      'Average treatment effect heavily driven by Sub-Saharan Africa'
    ],
    bugs: [],
    robustnessNotes: 'Dropping Sub-Saharan Africa eliminates statistical significance entirely (p=0.499), suggesting the global result is driven by one region.',
    hasRobustnessConcerns: true,
    images: ['figure1a.png','figure1b.png','figure2.png'],
    codeFiles: ['01_clean.py','02_figure1.py','03_tables.py','04_figure2.py','05_data_audit.py','06_robustness.py','utils.py']
  },
  {
    id: '228101',
    title: 'Gender Gaps in Entrepreneurship: Business Networks and Collaborations in Ghana',
    authors: 'Lambon-Quayefio, Asiedu, Truffa & Wong',
    journal: 'AEA Papers and Proceedings',
    year: 2025,
    topic: 'Development',
    verdict: 'replicated',
    mainClaim: 'Female business owners in Ghana have smaller business networks and lower profits, partially mediated by education, business location, and caregiving.',
    summaryFindings: 'All main coefficients, SEs, and significance levels consistent. Female business owners earn ~GHS 889 less/month (-30%). Adding product FE reduces the female coefficient by ~40%.',
    tablesReplicated: 'Tables 1, 2, 3; Figures 1a, 1b',
    keyFindings: [
      'Female business owners earn ~GHS 889 less per month (-30%) \u2014 confirmed',
      'Adding product FE reduces the female coefficient by ~40% and renders it insignificant',
      'All main coefficients, standard errors, and significance levels are consistent'
    ],
    notableIssues: [
      'Code bug: output.do references variable profit that is never created in clean.do',
      'Packaging oversight only \u2014 does not affect published results'
    ],
    bugs: [
      'output.do references a variable (profit) that is never created by the cleaning script (clean.do). Packaging oversight.'
    ],
    robustnessNotes: 'Quantile regression shows gender gap concentrated at median and above.',
    hasRobustnessConcerns: false,
    images: ['Figure1a.png','Figure1b.png'],
    codeFiles: ['01_clean.py','02_table1.py','03_figures.py','04_table3.py','05_table2.py','06_data_audit.py','07_robustness.py','utils.py']
  },
  {
    id: '237010',
    title: 'Temporary Layoffs, Loss-of-Recall, and Cyclical Unemployment Dynamics',
    authors: 'Gertler, Huckfeldt & Trigari',
    journal: 'American Economic Review',
    year: 2025,
    topic: 'Macroeconomics',
    verdict: 'replicated',
    mainClaim: 'Temporary layoffs with loss-of-recall amplify recessions; the 2020 COVID recession was dominated by temporary layoffs with rapid recall.',
    summaryFindings: 'All 12+ tables reproduce exactly from pre-computed statistics files. Transition matrix independently verified (15/16 entries within 0.002). COVID recession: 78.1% of unemployment increase from temporary layoffs.',
    tablesReplicated: 'Tables 1\u20132, 5, 7, 9\u201310, A.2\u2013A.3; Figures 1\u20132, A.5',
    keyFindings: [
      'All tables reproduce exactly from pre-computed sed-format files',
      'Transition matrix independently verified: 15/16 entries within 0.002',
      'COVID recession: 78.1% of unemployment increase from temporary layoffs',
      'Exceptionally well-organized replication package'
    ],
    notableIssues: [],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['figure_1_adjusted_unemployment.png','figure_2_covid_unemployment.png','figure_A5_sipp_hazards.png','figure_transition_probs.png','figure_covid_transitions.png','figure_covid_stocks.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '238484',
    title: 'The Price of War',
    authors: 'Federle, Meier, Muller, Mutschler & Schularick',
    journal: 'American Economic Review',
    year: 2025,
    topic: 'Macroeconomics',
    verdict: 'partial',
    mainClaim: 'Wars cause persistent GDP declines at war sites (~6\u20138% after 8 years), propagating through trade and proximity channels.',
    summaryFindings: 'Table 1 descriptive statistics match exactly (694 war sites, 502 wars). LP impulse responses show qualitatively identical patterns. Exact coefficients differ due to Driscoll-Kraay vs clustered SEs.',
    tablesReplicated: 'Table 1; Figures 2\u20136 (qualitative match)',
    keyFindings: [
      'Table 1 descriptives: 694 war sites, 502 wars \u2014 exact match',
      'LP impulse responses show qualitatively identical patterns to published figures',
      'Coefficient differences stem from Driscoll-Kraay SEs (not available in Python) vs cluster SEs'
    ],
    notableIssues: [
      'Driscoll-Kraay standard errors not available in Python; clustered SEs used as approximation'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['figure_2_casualty_barplot.png','figure_3a_casualty_histograms.png','figure_4_casroles_lgdp.png','figure_4_casroles_lcpi.png','figure_5_castrd_lgdp.png','figure_5_castrd_lcpi.png','figure_6_casprox_lgdp.png','figure_6_casprox_lcpi.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '238658',
    title: 'Negative Control Falsification Tests for Instrumental Variable Designs',
    authors: 'Danieli, Nevo & Oster',
    journal: 'Review of Economics and Statistics',
    year: 2025,
    topic: 'Econometrics',
    verdict: 'partial',
    mainClaim: 'Negative control falsification tests can detect IV validity violations that standard overidentification tests miss.',
    summaryFindings: 'Literature survey matches exactly (51%). ADH F-test and Bonferroni reject at <0.01. One p-value discrepancy for ADH (0.039 vs 0.403) due to cluster-robust SE differences between R and Python.',
    tablesReplicated: 'Tables 2, 4, 5; Figure 1',
    keyFindings: [
      'Literature survey: 51% \u2014 exact match',
      'ADH F-test and Bonferroni reject at p<0.01 \u2014 confirmed',
      'Ashraf-Galor reduced-form pattern matches',
      'Single NCO p-value discrepancy (0.039 vs 0.403) due to R vs Python cluster SE implementation'
    ],
    notableIssues: [
      'GAM tests not replicated (requires R mgcv package equivalent)',
      'Cluster-robust SE implementation differences between R and Python affect one p-value'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['Figure1_cor_cor_plot.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '239496',
    title: 'The Likelihood of Persistently Low Global Fertility',
    authors: 'Geruso & Spears',
    journal: 'Journal of Economic Perspectives',
    year: 2026,
    topic: 'Demography',
    verdict: 'replicated',
    mainClaim: 'Where completed cohort fertility has fallen below 1.9, no subsequent cohort has rebounded above 2.1 \u2014 a striking 0-for-24 record.',
    summaryFindings: '0-for-24 rebound statistic confirmed exactly. 67.1% of world population in TFR<2 countries. Kitagawa decomposition ~39%/61% confirmed.',
    tablesReplicated: 'Figures 1\u20133, 5a, 6, 7; Table 1, Table A2',
    keyFindings: [
      '0-for-24 countries with CCF rebounds \u2014 exact match of the central claim',
      '67.1% of world population in TFR<2 countries confirmed',
      'Kitagawa decomposition: ~39% within-country, ~61% between-country \u2014 confirmed'
    ],
    notableIssues: [
      'India DHS figures not replicated due to data access requirements (registration needed)'
    ],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['Figure1_births_spike.png','Figure2_TFR_sub2.png','Figure3_CCF_vs_TFR.png','Figure5a_CEB_parous.png','Figure6_latest_vs_lowest.png','Figure7_Romania_Bulgaria.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '239791',
    title: 'Family Institutions and the Global Fertility Transition',
    authors: 'Gobbi, Hannusch & Rossi',
    journal: 'Journal of Economic Perspectives',
    year: 2026,
    topic: 'Demography',
    verdict: 'replicated',
    mainClaim: 'Countries with good family institutions (monogamy, partible inheritance) show much stronger correlations between development indicators and fertility decline.',
    summaryFindings: 'All 24 figures match exactly. R-squared values match for all Figure 4 specifications (mono: 0.816, non-mono: 0.245). Minor coding bug found: GDP differences computed in levels rather than logs.',
    tablesReplicated: 'Figures 1a\u2013d, 2a\u2013d, 4a\u2013d, A1a\u2013f, A2a\u2013f (24 total)',
    keyFindings: [
      'All 24 figures match exactly',
      'R-squared: monogamous countries 0.816 vs non-monogamous 0.245',
      'GDP coding bug: differences computed in levels rather than logs despite variable name dlgdppc',
      'Bug changes R-squared gap from 0.571 to 0.519 but does not affect qualitative conclusions'
    ],
    notableIssues: [
      'Small Sub-Saharan Africa subgroup samples (N=16, 23) limit inference'
    ],
    bugs: [
      'GDP differences computed in levels rather than logs despite the variable being named dlgdppc (delta-log-GDP-per-capita). Changes R-squared gap from 0.571 to 0.519.'
    ],
    robustnessNotes: 'Level-vs-log bug modestly affects the magnitude of the R-squared gap between institutional types but does not change the qualitative story.',
    hasRobustnessConcerns: true,
    images: ['Figure1a.png','Figure1b.png','Figure1c.png','Figure1d.png','Figure2a.png','Figure2b.png','Figure2c.png','Figure2d.png','Figure4a.png','Figure4b.png','Figure4c.png','Figure4d.png','FigureA1a.png','FigureA1b.png','FigureA1c.png','FigureA1d.png','FigureA1e.png','FigureA1f.png','FigureA2a.png','FigureA2b.png','FigureA2c.png','FigureA2d.png','FigureA2e.png','FigureA2f.png'],
    codeFiles: ['01_clean.py','02_figures.py','03_tables.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '241085',
    title: 'Antitrust Enforcement in Labor Markets',
    authors: 'Prager',
    journal: 'Journal of Economic Perspectives',
    year: 2026,
    topic: 'Labor',
    verdict: 'replicated',
    mainClaim: 'Occupation-based labor market definitions using worker transition probabilities provide more meaningful delineation than standard classification codes.',
    summaryFindings: '8-occupation Production Clerk cluster matches exactly. Weighted-average transition correlation of 0.058 replicates exactly. 49% of clusters cross SOC major-group boundaries.',
    tablesReplicated: 'Figure 1 (dendrogram)',
    keyFindings: [
      '8-occupation Production Clerk cluster matches exactly',
      'Weighted-average transition correlation: 0.058 \u2014 exact match',
      '49% of clusters cross SOC major-group boundaries \u2014 confirmed',
      'Minimal but complete replication package'
    ],
    notableIssues: [],
    bugs: [],
    robustnessNotes: '',
    hasRobustnessConcerns: false,
    images: ['Figure1_dendrogram.png','Figure_full_dendrogram.png'],
    codeFiles: ['01_clean.py','02_tables.py','03_figures.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '199083',
    title: 'Moonshot: Public R&D and Growth',
    authors: 'Kantor & Whalley',
    journal: 'American Economic Review',
    year: 2023,
    topic: 'Public Economics',
    verdict: 'replicated',
    mainClaim: 'Space Race-era public R&D spending generated positive local manufacturing spillovers, concentrated in NASA-related industries in counties with high space capability.',
    summaryFindings: 'Core triple-difference result replicates: NASA industry × high space capability × post-Space Race coefficient is +0.345*** for value added and +0.388*** for employment. Non-NASA industry placebo coefficients are near zero. N=4,302 (49 counties).',
    tablesReplicated: 'Tables 1, 2A, 3',
    keyFindings: [
      'DDD coefficient +0.345*** (SE=0.112) for log value added — significant at 1%',
      'DDD coefficient +0.388*** (SE=0.120) for log employment — significant at 1%',
      'First stage confirmed: NASA spending and patents strongly predicted by treatment',
      'Non-NASA industry placebo coefficients near zero, supporting DDD identification'
    ],
    notableIssues: [
      'Sample size (4,302 obs, 49 counties) smaller than paper due to county crosswalk and MSA fill-in differences',
      '1967 has notably fewer observations (61 vs ~475) due to MSA fill-in coverage'
    ],
    bugs: [],
    robustnessNotes: 'Highly stable across 12 robustness checks including dropping years, alternative outcomes, clustering alternatives, and FE specifications.',
    hasRobustnessConcerns: false,
    images: [],
    codeFiles: ['01_build.py','02_tables.py','04_data_audit.py','05_robustness.py','utils.py']
  },
  {
    id: '206261',
    title: 'Immigration, Innovation, and Growth',
    authors: 'Burchardi, Chaney, Hassan, Tarquinio & Terry',
    journal: 'American Economic Review',
    year: 2024,
    topic: 'Labor',
    verdict: 'replicated',
    mainClaim: 'Immigration causally increases local innovation (patenting) and wages, identified using a three-stage shift-share instrument based on historical ancestry and push-pull migration drivers.',
    summaryFindings: 'IV coefficient on patents is positive and marginally significant (+1.03, t=1.66) with very strong first stage (F=803). Wage effect is significant (+0.25, t=2.32). Results stable across sample restrictions. 21,987 obs, 3,141 counties, 7 periods.',
    tablesReplicated: 'Tables 1-4',
    keyFindings: [
      'IV (2SLS) patent coefficient: +1.03* (SE=0.62), consistent with paper',
      'Very strong first stage: F=803, instrument-endogenous correlation=0.86',
      'Wage IV: +0.25** (SE=0.11, t=2.32) — immigration raises local wages',
      'OLS (+1.29) and IV (+1.03) are similar, consistent with measurement error attenuation'
    ],
    notableIssues: [
      'Patent outcome has 40% missing data (available for 5 of 7 periods only)',
      'County FE IV specification numerically unstable in our implementation',
      'Three-stage instrument involves ~20 Stata/MATLAB/R scripts processing ~50 GB of raw data'
    ],
    bugs: [],
    robustnessNotes: 'IV coefficient stable (0.81-1.19) across leave-one-period-out checks. Alternative patent measures give consistent results. County FE specification unstable.',
    hasRobustnessConcerns: false,
    images: [],
    codeFiles: ['01_build.py','02_tables.py','04_data_audit.py','05_robustness.py','run_from_phase2.py','utils.py']
  }
];


// =============================================================================
// RENDERING FUNCTIONS
// =============================================================================

function renderSummary() {
  const total = STUDIES.length;
  const replicated = STUDIES.filter(s => s.verdict === 'replicated').length;
  const partial = STUDIES.filter(s => s.verdict === 'partial').length;
  const bugsFound = STUDIES.filter(s => s.bugs.length > 0).length;
  const concerns = STUDIES.filter(s => s.hasRobustnessConcerns).length;
  const repPct = Math.round(replicated / total * 100);
  const partPct = Math.round(partial / total * 100);

  document.getElementById('stats-bar').innerHTML = `
    <div class="stat-card">
      <div class="number">${total}</div>
      <div class="label">Papers Examined</div>
    </div>
    <div class="stat-card replicated">
      <div class="number">${replicated}</div>
      <div class="label">Fully Replicated (${repPct}%)</div>
    </div>
    <div class="stat-card partial">
      <div class="number">${partial}</div>
      <div class="label">Partially Replicated (${partPct}%)</div>
    </div>
    <div class="stat-card bugs">
      <div class="number">${bugsFound}</div>
      <div class="label">Coding Bugs Discovered</div>
    </div>
  `;

  document.getElementById('summary-content').innerHTML = `
    <h2>About This Project</h2>
    <p>
      This project uses AI (Claude by Anthropic) to systematically replicate published economics papers
      from the American Economic Review family of journals. Each replication follows a standardized pipeline:
      data cleaning, table reproduction, figure reproduction, comprehensive data auditing, and robustness
      testing that goes beyond the original paper's scope.
    </p>
    <p>
      All ${total} replications were performed using Python with pandas and statsmodels, translating
      original Stata and R code independently and verifying every published coefficient, standard error,
      and sample size. Each study also includes original robustness checks not found in the published paper.
    </p>

    <div class="outcome-bar">
      <div class="outcome-segment replicated" style="width:${repPct}%">${replicated} Replicated</div>
      <div class="outcome-segment partial" style="width:${partPct}%">${partial} Partial</div>
    </div>

    <h2>What We Found</h2>
    <p>
      Of ${total} papers examined, <strong>${replicated} (${repPct}%)</strong> were fully replicated &mdash;
      every major table and figure matched the published results within rounding tolerance.
      <strong>${partial} papers (${partPct}%)</strong> were partially replicated.
      <strong>No paper completely failed to replicate.</strong>
    </p>

    <h2>Why Some Replications Are Partial</h2>
    <p>
      The ${partial} partial replications fall into three categories with very different
      implications. We think it's important to be transparent about what's on us vs. what isn't.
    </p>
    ${renderPartialTable()}

    <h2>Coding Bugs Discovered</h2>
    <p>
      AI-driven replication uncovered coding bugs in <strong>${bugsFound} papers (${Math.round(bugsFound/total*100)}%)</strong>:
    </p>
    <ul>
      ${STUDIES.filter(s => s.bugs.length > 0).map(s => `
        <li><strong><a href="#study/${s.id}">${s.authors} (${s.year})</a>:</strong> ${s.bugs[0]}</li>
      `).join('')}
    </ul>
    <p>
      None of these bugs changed the qualitative conclusions of the papers, but they represent
      meaningful quantitative discrepancies that traditional peer review did not catch.
    </p>

    <h2>Robustness Concerns</h2>
    <p>
      Beyond bugs, AI robustness testing identified substantive concerns in
      <strong>${concerns} papers (${Math.round(concerns/total*100)}%)</strong> &mdash;
      cases where the original findings are more fragile than presented:
    </p>

    <div class="highlight-grid">
      <div class="highlight-card">
        <h3>Subgroup Fragility</h3>
        <p>Treatment effects driven by a single subgroup or region that, when excluded, eliminate statistical significance.</p>
        <ul class="concern-papers">
          <li><a href="#study/112786">Rainfall Forecasts &amp; Wages</a> &mdash; migration result driven by a single village</li>
          <li><a href="#study/112801">Declining Fortunes of the Young</a> &mdash; decline is entirely a male phenomenon</li>
          <li><a href="#study/112802">Earnings Differentials Across Majors</a> &mdash; widening trend driven entirely by older workers</li>
          <li><a href="#study/112819">Global Fuel Subsidies</a> &mdash; two countries account for 50% of deadweight loss</li>
          <li><a href="#study/227802">Childcare Laws &amp; Women's Empowerment</a> &mdash; dropping Sub-Saharan Africa eliminates significance (p=0.499)</li>
        </ul>
      </div>
      <div class="highlight-card">
        <h3>Measure Sensitivity</h3>
        <p>Results that change sign or significance when using alternative but equally valid variable definitions or specifications.</p>
        <ul class="concern-papers">
          <li><a href="#study/112783">Killer Apps in the iPhone Ecosystem</a> &mdash; Number of Versions effect loses significance dropping outliers</li>
          <li><a href="#study/112803">Return of the Solow Paradox?</a> &mdash; alternative IT measure vintage reverses coefficient sign</li>
          <li><a href="#study/112875">Election Fraud in Afghanistan</a> &mdash; vote-stuffing result insignificant after winsorizing at 1%</li>
          <li><a href="#study/112876">Credit Crisis &amp; Farm Land Prices</a> &mdash; panel specification fragile to dropping census years</li>
          <li><a href="#study/179162">Mother-Father Spending Differences</a> &mdash; gender differential disappears for incentivized goods and follow-up wave</li>
        </ul>
      </div>
      <div class="highlight-card">
        <h3>Identification Concerns</h3>
        <p>Failed placebo tests or pre-treatment violations suggesting potential issues with identifying assumptions.</p>
        <ul class="concern-papers">
          <li><a href="#study/112792">Disability Insurance &amp; Health Reform</a> &mdash; pre-treatment event-study coefficients and low-power permutation test</li>
          <li><a href="#study/112799">Race and Marriage in the Labor Market</a> &mdash; failed placebo test suggests block-randomization correlations may bias estimates</li>
        </ul>
      </div>
      <div class="highlight-card">
        <h3>Trend-Driven Correlations</h3>
        <p>Statistical relationships driven entirely by shared secular trends rather than causal mechanisms.</p>
        <ul class="concern-papers">
          <li><a href="#study/112818">Policy Uncertainty Since 1960</a> &mdash; detrended and first-differenced correlations are near zero</li>
        </ul>
      </div>
      <div class="highlight-card">
        <h3>Bug-Amplified Results</h3>
        <p>Code bugs that meaningfully affect the magnitude of reported estimates, even when qualitative conclusions survive.</p>
        <ul class="concern-papers">
          <li><a href="#study/192297">Big Loans to Small Businesses</a> &mdash; coding bug inflates average treatment effect by ~40%</li>
          <li><a href="#study/239791">Family Institutions &amp; Fertility</a> &mdash; level-vs-log bug affects R-squared gap magnitude</li>
          <li><a href="#study/112820">Power Plants in India</a> &mdash; multiple code bugs compounded by declining data reporting rates</li>
        </ul>
      </div>
    </div>

    <div class="robust-passed">
      <strong>Papers that passed robustness checks with no concerns:</strong>
      The remaining ${total - concerns} papers showed no robustness issues. Several were notably stable:
      <ul>
        <li><a href="#study/112323">Entry, Exit &amp; Investment-Specific Technical Change</a> &mdash; robust across 34 specifications (placebo p=0.001)</li>
        <li><a href="#study/112805">School Choice Lotteries &amp; School Effectiveness</a> &mdash; all 19 robustness checks confirm findings</li>
        <li><a href="#study/112874">Too-Systemic-To-Fail</a> &mdash; survives alternative deltas, crisis dates, winsorization, leave-one-sector-out, and placebo spreads</li>
        <li><a href="#study/199083">Moonshot: Public R&amp;D and Growth</a> &mdash; stable across 12 checks including alternative outcomes and FE specifications</li>
        <li><a href="#study/206261">Immigration, Innovation, and Growth</a> &mdash; IV coefficient stable (0.81&ndash;1.19) across leave-one-period-out checks</li>
      </ul>
    </div>

    <p>
      These findings don't invalidate the original papers but provide important context for
      interpreting their results. Click into any study below to see the full replication details,
      figures, and code.
    </p>

    <h2>Results by Publication Year</h2>
    <p>
      Do older papers have more issues? The table below groups papers by era and
      compares replication outcomes, coding bugs, and robustness concerns. Issue rate
      counts papers with <em>at least one</em> bug or robustness concern (no double-counting).
    </p>
    ${renderYearTable()}
  `;
}


function renderYearTable() {
  const groups = [
    { label: '2005\u20132008', range: [2005, 2008] },
    { label: '2014\u20132016', range: [2014, 2016] },
    { label: '2022\u20132024', range: [2022, 2024] },
    { label: '2025\u20132026', range: [2025, 2026] },
  ];

  function stats(papers) {
    const total = papers.length;
    const replicated = papers.filter(s => s.verdict === 'replicated').length;
    const partial = papers.filter(s => s.verdict === 'partial').length;
    const bugs = papers.filter(s => s.bugs.length > 0).length;
    const concerns = papers.filter(s => s.hasRobustnessConcerns).length;
    const withAnyIssue = papers.filter(s => s.bugs.length > 0 || s.hasRobustnessConcerns).length;
    const issueRate = total > 0 ? Math.round(withAnyIssue / total * 100) : 0;
    return { total, replicated, partial, bugs, concerns, withAnyIssue, issueRate };
  }

  const groupData = groups.map(g => {
    const papers = STUDIES.filter(s => s.year >= g.range[0] && s.year <= g.range[1]);
    return { ...g, ...stats(papers) };
  });

  const allStats = stats(STUDIES);

  const rows = groupData.map(g => `
    <tr>
      <td class="cell-label">${g.label}</td>
      <td class="cell-num">${g.total}</td>
      <td class="cell-num cell-replicated">${g.replicated}</td>
      <td class="cell-num cell-partial">${g.partial}</td>
      <td class="cell-num cell-bug">${g.bugs}</td>
      <td class="cell-num cell-concern">${g.concerns}</td>
      <td class="cell-num cell-rate">${g.issueRate}%</td>
    </tr>
  `).join('');

  // Takeaway
  const pre = stats(STUDIES.filter(s => s.year <= 2016));
  const post = stats(STUDIES.filter(s => s.year >= 2022));

  let takeaway;
  if (pre.issueRate > post.issueRate) {
    takeaway = `Pre-2020 papers have a higher issue rate (${pre.issueRate}%) than post-2020 papers (${post.issueRate}%). The 2014\u20132016 era has the most robustness concerns, partly reflecting the large batch of shorter-format AEA P&amp;P papers. Bugs appear across all eras.`;
  } else if (pre.issueRate === post.issueRate) {
    takeaway = `Issue rates are comparable across eras (${pre.issueRate}% each). Bugs and robustness concerns appear across all publication years with no clear trend.`;
  } else {
    takeaway = `Post-2020 papers show a slightly higher issue rate (${post.issueRate}%) vs pre-2020 (${pre.issueRate}%), though small sample sizes make it hard to draw firm conclusions.`;
  }

  return `
    <table class="year-table">
      <thead>
        <tr>
          <th>Years</th>
          <th>Papers</th>
          <th>Replicated</th>
          <th>Partial</th>
          <th>Bugs</th>
          <th>Robustness Concerns</th>
          <th>Issue Rate</th>
        </tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
      <tfoot>
        <tr>
          <td class="cell-label"><strong>Total</strong></td>
          <td class="cell-num"><strong>${allStats.total}</strong></td>
          <td class="cell-num cell-replicated"><strong>${allStats.replicated}</strong></td>
          <td class="cell-num cell-partial"><strong>${allStats.partial}</strong></td>
          <td class="cell-num cell-bug"><strong>${allStats.bugs}</strong></td>
          <td class="cell-num cell-concern"><strong>${allStats.concerns}</strong></td>
          <td class="cell-num cell-rate"><strong>${allStats.issueRate}%</strong></td>
        </tr>
      </tfoot>
    </table>
    <p style="margin-top:1rem;">${takeaway}</p>
  `;
}


function renderPartialTable() {
  // Categorize each partial replication by reason
  const partials = [
    {
      category: 'Restricted Data',
      categoryNote: 'By design \u2014 not a replication failure. Some papers use proprietary or confidential data that cannot be shared publicly.',
      cssClass: 'cat-restricted',
      studies: [
        { id: '221423', authors: 'Mogstad, Salvanes & Torsvik', reason: '11 of 14 tables require PIAAC microdata (OECD registration)' },
        { id: '225841', authors: 'Greenwald & Guren', reason: 'Published results use confidential GG microdata; only pseudodata available' }
      ]
    },
    {
      category: 'Missing Tool',
      categoryNote: 'On us \u2014 we should have written an equivalent or found an alternative implementation in Python.',
      cssClass: 'cat-tool',
      studies: [
        { id: '227802', authors: 'Anukriti et al.', reason: 'synthdid R package; used TWFE approximation instead' },
        { id: '238484', authors: 'Federle et al.', reason: 'Driscoll-Kraay SEs; used clustered SEs as approximation' },
        { id: '238658', authors: 'Danieli, Nevo & Oster', reason: 'GAM tests (R mgcv package); no Python equivalent used' }
      ]
    },
    {
      category: 'Package Issue',
      categoryNote: 'On the original authors \u2014 the replication package shipped with missing files, no code, or data discrepancies.',
      cssClass: 'cat-package',
      studies: [
        { id: '112319', authors: 'Charness & Levin', reason: 'No analysis code provided; all code reconstructed from paper' },
        { id: '112820', authors: 'Chan, Cropper & Malik', reason: 'Code bugs: references to nonexistent variables in matching scripts' },
        { id: '173341', authors: 'Bobonis et al.', reason: 'Shipped data has different values than published .tex outputs' },
        { id: '208367', authors: 'Andreolli & Surico', reason: '3 MATLAB output files missing from package' }
      ]
    }
  ];

  const rows = partials.map(cat => {
    const studyRows = cat.studies.map((s, i) => `
      <tr class="${cat.cssClass}">
        ${i === 0 ? `<td class="cell-label cat-cell" rowspan="${cat.studies.length}">
          <strong>${cat.category}</strong><br>
          <span class="cat-note">${cat.categoryNote}</span>
        </td>` : ''}
        <td><a href="#study/${s.id}">${s.authors}</a></td>
        <td>${s.reason}</td>
      </tr>
    `).join('');
    return studyRows;
  }).join('');

  return `
    <table class="partial-table">
      <thead>
        <tr>
          <th style="width:28%">Category</th>
          <th style="width:22%">Paper</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>
    <p style="margin-top:0.75rem;">
      Of the ${partials.reduce((a, c) => a + c.studies.length, 0)} partial replications:
      <strong>${partials[0].studies.length}</strong> are due to restricted data (not our fault),
      <strong>${partials[1].studies.length}</strong> are missing tools (our fault \u2014 we should have built them),
      and <strong>${partials[2].studies.length}</strong> are replication package issues (on the original authors).
    </p>
  `;
}


function renderFilterBar() {
  const topics = [...new Set(STUDIES.map(s => s.topic))].sort();
  const journals = [...new Set(STUDIES.map(s => s.journal))].sort();

  document.getElementById('filter-bar').innerHTML = `
    <input type="text" id="search" class="search-input" placeholder="Search by title, author, or topic...">
    <select id="verdict-filter" class="filter-select">
      <option value="">All Verdicts</option>
      <option value="replicated">Fully Replicated</option>
      <option value="partial">Partially Replicated</option>
    </select>
    <select id="topic-filter" class="filter-select">
      <option value="">All Topics</option>
      ${topics.map(t => `<option value="${t}">${t}</option>`).join('')}
    </select>
    <select id="journal-filter" class="filter-select">
      <option value="">All Journals</option>
      ${journals.map(j => `<option value="${j}">${j}</option>`).join('')}
    </select>
  `;

  document.getElementById('search').addEventListener('input', applyFilters);
  document.getElementById('verdict-filter').addEventListener('change', applyFilters);
  document.getElementById('topic-filter').addEventListener('change', applyFilters);
  document.getElementById('journal-filter').addEventListener('change', applyFilters);
}


function applyFilters() {
  const query = document.getElementById('search').value.toLowerCase();
  const verdict = document.getElementById('verdict-filter').value;
  const topic = document.getElementById('topic-filter').value;
  const journal = document.getElementById('journal-filter').value;

  const filtered = STUDIES.filter(s => {
    if (verdict && s.verdict !== verdict) return false;
    if (topic && s.topic !== topic) return false;
    if (journal && s.journal !== journal) return false;
    if (query) {
      const hay = (s.title + ' ' + s.authors + ' ' + s.topic + ' ' + s.mainClaim + ' ' + s.id).toLowerCase();
      if (!hay.includes(query)) return false;
    }
    return true;
  });

  renderGrid(filtered);
}


function renderGrid(studies) {
  const grid = document.getElementById('study-grid');
  if (studies.length === 0) {
    grid.innerHTML = '<div class="no-results">No studies match your filters.</div>';
    return;
  }

  grid.innerHTML = studies.map(s => {
    const vc = s.verdict === 'replicated' ? 'badge-replicated' : s.verdict === 'partial' ? 'badge-partial' : 'badge-failed';
    const vl = s.verdict === 'replicated' ? 'Replicated' : s.verdict === 'partial' ? 'Partial' : 'Failed';

    return `
      <a href="#study/${s.id}" class="study-card">
        <div class="card-header">
          <span class="badge ${vc}">${vl}</span>
          <span class="card-year">${s.year}</span>
        </div>
        <h3 class="card-title">${s.title}</h3>
        <p class="card-authors">${s.authors}</p>
        <p class="card-journal">${s.journal}</p>
        <p class="card-summary">${s.mainClaim}</p>
        <div class="card-flags">
          ${s.bugs.length > 0 ? '<span class="badge badge-bug">Bug Found</span>' : ''}
          ${s.hasRobustnessConcerns ? '<span class="badge badge-concern">Robustness Concerns</span>' : ''}
        </div>
      </a>
    `;
  }).join('');
}


function renderDetail(study) {
  const vc = study.verdict === 'replicated' ? 'badge-replicated' : study.verdict === 'partial' ? 'badge-partial' : 'badge-failed';
  const vl = study.verdict === 'replicated' ? 'Fully Replicated' : study.verdict === 'partial' ? 'Partially Replicated' : 'Failed to Replicate';
  const icpsrUrl = `${CONFIG.openicpsrBase}/${study.id}/version/V1/view`;
  const repoBase = `${CONFIG.repoUrl}/blob/main/replication_${study.id}`;

  return `
    <div class="detail-view">
      <a href="#" class="back-btn">\u2190 Back to all studies</a>

      <div class="detail-header">
        <span class="badge ${vc}">${vl}</span>
        <h1>${study.title}</h1>
        <p class="detail-meta">${study.authors} (${study.year})</p>
        <p class="detail-meta"><em>${study.journal}</em> &middot; ${study.topic}</p>
        <div class="detail-links">
          <a href="${icpsrUrl}" target="_blank">\u{1F4E6} Official Reproduction Package</a>
          <a href="${repoBase}" target="_blank">\u{1F4BB} Replication Code</a>
        </div>
      </div>

      <div class="detail-section">
        <h2>Paper's Main Claim</h2>
        <p>${study.mainClaim}</p>
      </div>

      <div class="detail-section">
        <h2>Replication Results</h2>
        <p>${study.summaryFindings}</p>
        <p><strong>Tables &amp; Figures Replicated:</strong> ${study.tablesReplicated}</p>
      </div>

      ${study.bugs.length > 0 ? `
      <div class="detail-section bugs-section">
        <h2>Coding Bugs Found</h2>
        <ul>${study.bugs.map(b => '<li>' + b + '</li>').join('')}</ul>
      </div>
      ` : ''}

      <div class="detail-section">
        <h2>Key Findings</h2>
        <ul>${study.keyFindings.map(f => '<li>' + f + '</li>').join('')}</ul>
      </div>

      ${study.notableIssues.length > 0 ? `
      <div class="detail-section">
        <h2>Notable Issues</h2>
        <ul>${study.notableIssues.map(i => '<li>' + i + '</li>').join('')}</ul>
      </div>
      ` : ''}

      ${study.robustnessNotes ? `
      <div class="detail-section ${study.hasRobustnessConcerns ? 'concerns-section' : ''}">
        <h2>Robustness Assessment</h2>
        <p>${study.robustnessNotes}</p>
      </div>
      ` : ''}

      ${study.images.length > 0 ? `
      <div class="detail-section">
        <h2>Replicated Figures</h2>
        <p style="color:var(--text-secondary);font-size:0.875rem;margin-bottom:1rem;">Click any figure to enlarge.</p>
        <div class="image-gallery">
          ${study.images.map(img => {
            const caption = img.replace(/\.(png|jpg|jpeg|gif)$/i, '').replace(/[_-]/g, ' ');
            return `
              <div class="image-item">
                <img src="images/${study.id}/${img}" alt="${caption}" loading="lazy" onclick="openLightbox(this.src)">
                <p class="image-caption">${caption}</p>
              </div>
            `;
          }).join('')}
        </div>
      </div>
      ` : ''}

      <div class="detail-section">
        <h2>Replication Code</h2>
        <div class="code-links">
          ${study.codeFiles.map(f =>
            `<a href="${repoBase}/${f}" target="_blank" class="code-link">${f}</a>`
          ).join('')}
        </div>
      </div>
    </div>
  `;
}


// =============================================================================
// LIGHTBOX
// =============================================================================

function openLightbox(src) {
  const lb = document.createElement('div');
  lb.className = 'lightbox';
  lb.innerHTML = `<img src="${src}" alt="Enlarged figure">`;
  lb.addEventListener('click', function() { lb.remove(); });
  document.body.appendChild(lb);
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    const lb = document.querySelector('.lightbox');
    if (lb) lb.remove();
  }
});


// =============================================================================
// ROUTING
// =============================================================================

function handleRoute() {
  const hash = window.location.hash;

  if (hash.startsWith('#study/')) {
    const id = hash.split('/')[1];
    const study = STUDIES.find(s => s.id === id);
    if (study) {
      showDetail(study);
      return;
    }
  }

  showGrid();
}

function showDetail(study) {
  document.getElementById('summary').classList.add('hidden');
  document.getElementById('studies-section').classList.add('hidden');
  const detail = document.getElementById('detail-section');
  detail.classList.remove('hidden');
  detail.innerHTML = renderDetail(study);
  window.scrollTo(0, 0);
}

function showGrid() {
  document.getElementById('summary').classList.remove('hidden');
  document.getElementById('studies-section').classList.remove('hidden');
  document.getElementById('detail-section').classList.add('hidden');
}


// =============================================================================
// INIT
// =============================================================================

function init() {
  renderSummary();
  renderFilterBar();
  renderGrid(STUDIES);
  handleRoute();
  window.addEventListener('hashchange', handleRoute);
}

document.addEventListener('DOMContentLoaded', init);
