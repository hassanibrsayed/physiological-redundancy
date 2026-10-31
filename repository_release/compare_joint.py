"""Compare the Bonferroni-split thresholds (Pilot 9 / Pilot 11) with the joint thresholds (Pilot 12 / Pilot 13).

Run after run_joint.ps1:  .\\.venv\\Scripts\\python.exe compare_joint.py
Reads pilot9_comb_*.csv, pilot12_comb_*.csv, pilot11_cost.csv, pilot13_cost.csv. Writes nothing."""
import pandas as pd

pd.set_option('display.width', 180)


def grid_cost(prefix):
    c = pd.read_csv(f'{prefix}_comb_cost.csv')
    c = c[c.detector == 'Redundancy']
    ad = c.groupby(['case', 'target', 'mag']).cost.min().reset_index()
    return ad.groupby('target').agg(evasion_found=('cost', lambda x: x.notna().mean()), mean_cost=('cost', 'mean')).round(2)


def red_cost(prefix, detector='with C4', mode='timing-free'):
    c = pd.read_csv(f'{prefix}_cost.csv')
    c = c[(c.detector == detector) & (c['mode'] == mode)]
    return c.groupby('target').agg(evasion_found=('cost', lambda x: x.notna().mean()), mean_cost=('cost', 'mean')).round(2)


def det_by_k(prefix):
    d = pd.read_csv(f'{prefix}_comb_detection.csv')
    return d[d.detector == 'Redundancy'].groupby(['target', 'k']).detection.mean().round(2)


print('Grid attack cost against the redundancy detector (adaptive profile): Bonferroni split vs joint thresholds')
print(pd.concat({'Bonferroni (Pilot 9)': grid_cost('pilot9'), 'Joint (Pilot 12)': grid_cost('pilot12')}, axis=1).to_string())
print('\nDetection at matched 10% FPR by number of compromised devices (redundancy detector)')
print(pd.concat({'Bonferroni': det_by_k('pilot9'), 'Joint': det_by_k('pilot12')}, axis=1).to_string())
for det in ('with C4', 'without C4'):
    print(f'\nRed-team cost, detector {det}, timing-free: Bonferroni (Pilot 11) vs joint (Pilot 13)')
    print(pd.concat({'Bonferroni': red_cost('pilot11', det), 'Joint': red_cost('pilot13', det)}, axis=1).to_string())
