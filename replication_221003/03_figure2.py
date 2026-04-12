"""03_figure2.py — Replicate Figure 2 / Table A5: ITT impacts on cognitive and
social-emotional skills.

Spec: areg y treatment_ecd [pw=equal_weights] if istarget35==1,
      absorb(DIST_ID) vce(cluster COM_ID)
"""
import os
import numpy as np
import pandas as pd
from utils import load_endline, areg_cluster, OUTPUT_DIR

SKILL_VARS = [
    ('Preliteracy_zi', 'Early Literacy Skills'),
    ('Premath_zi', 'Early Math Skills'),
    ('Executive_funct_zi', 'Executive Function'),
    ('Socioemo_cogn_zi', 'Social-Emotional Development'),
    ('Fine_motor_zi', 'Fine Motor Skill for Writing'),
    ('literacyinterest_zi', 'Literacy interest'),
    ('skills_zi', 'Skills index'),
]


def main():
    el = load_endline()
    d = el[el['istarget35'] == 1].copy()

    rows = []
    for v, label in SKILL_VARS:
        r = areg_cluster(d, v, ['treatment_ecd'], 'DIST_ID', 'COM_ID',
                         weight_col='equal_weights')
        ctrl_mean = d[d['treatment_ecd'] == 0][v].mean()
        rows.append({'outcome': label, 'var': v,
                     'beta': r['beta'], 'se': r['se'],
                     'pval': r['pval'], 'N': r['N'],
                     'ctrl_mean': ctrl_mean})
    tab = pd.DataFrame(rows)
    print("\n" + "=" * 64)
    print("TABLE A5 / FIGURE 2: ITT Impacts on Child Cognitive / SE Skills")
    print("=" * 64)
    print(tab.to_string(index=False, float_format='%.4f'))
    tab.to_csv(os.path.join(OUTPUT_DIR, 'tableA5.csv'), index=False)


if __name__ == '__main__':
    main()
