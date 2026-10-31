"""Replication: the 8 design cases (joint thresholds: pilot12 / pilot13) versus the 12 unused sample cases
(pilot14 / pilot15). Every design decision was taken on the 8 cases, so agreement on the 12 is out-of-sample support.

Run after run_replication.ps1:  .\\.venv\\Scripts\\python.exe compare_replication.py
Reads pilot12_comb_*.csv, pilot14_comb_*.csv, pilot13_cost.csv, pilot15_cost.csv. Writes nothing."""
import pandas as pd

pd.set_option('display.width', 200)
LABELS = {'original': ('pilot12', 'pilot13'), 'replication': ('pilot14', 'pilot15')}
DETS = ['Redundancy', 'Learned', 'Split OR (5%+5%)']


def grid_cost(prefix):
    c = pd.read_csv(f'{prefix}_comb_cost.csv')
    ad = c.groupby(['detector', 'case', 'target', 'mag']).cost.min().reset_index()
    t = ad.groupby(['detector', 'target']).agg(cases=('case', 'nunique'), evasion=('cost', lambda x: x.notna().mean()),
                                               cost=('cost', 'mean')).round(2)
    return t.loc[DETS]


def det_by_k(prefix):
    d = pd.read_csv(f'{prefix}_comb_detection.csv')
    d = d[d.detector.isin(['Redundancy', 'Learned'])]
    return d.groupby(['target', 'k', 'detector']).detection.mean().unstack('detector').round(2)


def red_cost(prefix, detector):
    c = pd.read_csv(f'{prefix}_cost.csv')
    c = c[(c.detector == detector) & (c['mode'] == 'timing-free')]
    return c.groupby('target').agg(cases=('case', 'nunique'), evasion=('cost', lambda x: x.notna().mean()),
                                   cost=('cost', 'mean')).round(2)


print('Grid attack cost against each detector (attacker free to choose the profile): evasion found / mean cost')
print(pd.concat({k: grid_cost(v[0]) for k, v in LABELS.items()}, axis=1).to_string())
print('\nDetection at matched 10% FPR by number of compromised devices')
print(pd.concat({k: det_by_k(v[0]) for k, v in LABELS.items()}, axis=1).to_string())
for det in ('with C4', 'without C4'):
    print(f'\nRed-team cost, detector {det}, timing-free')
    print(pd.concat({k: red_cost(v[1], det) for k, v in LABELS.items()}, axis=1).to_string())
