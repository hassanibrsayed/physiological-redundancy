"""Pilot 17: a fusion designed for arterial pressure only.

Motivation. For arterial pressure the cross-device redundancy detector has little to compare (one arterial source in
most cases), and the tuned learned detectors beat it once several sources are compromised (Pilot 16: 0.42 against
0.28 at three sources in the design cases, 0.41 against 0.26 at two sources in the replication cases). For EtCO2,
heart rate and respiratory rate the learned scores added almost nothing. So we fuse the two kinds of detector for
arterial pressure only and look at how the split of the false-alarm budget matters.

Detectors (arterial-pressure windows only; evidence z = -log p from each case's clean control windows, as in
pilot8_combined.py; tuned learned detectors from pilot12tuned / pilot14tuned):
  Redundancy       invariants R2b, R3, C1-C4 (joint thresholds)
  Learned          maximum evidence of the tuned Isolation Forest, autoencoder and LSTM
  LSTM             the tuned LSTM alone
  Max evidence     maximum of Redundancy and Learned
  OR a%+b%         alarm if Redundancy passes its own cut at a% false alarms or Learned passes its cut at b%
                   (a+b = 10%, ties randomised independently); four splits are shown.
Every split is reported; none is selected on the test windows, and the 5%+5% split is the pre-specified one.
All thresholds are matched cuts on the clean test controls of all cases, so this is a descriptive comparison, and
the cost is a grid cost (smallest source set whose detection is at most the 10% false-alarm rate plus 10 points),
not a red-team cost: a red team against the fused detector has not been run.

Usage:  .\\.venv\\Scripts\\python.exe pilot17_bp_fusion.py       (needs pilot12tuned_base_scores.csv and pilot14tuned_base_scores.csv)
Output: pilot17_bp_fusion_detection.csv and printed tables."""
import numpy as np, pandas as pd

SETS = {'8 design cases': 'pilot12tuned', '12 replication cases': 'pilot14tuned'}
TARGET_FPR, MARGIN = 0.10, 0.10
BASE = {'Redundancy': 'red_target_score', 'IF': 'score_IF_all', 'AE': 'score_AE', 'LSTM': 'score_LSTM'}
SPLITS = [0.075, 0.05, 0.025, 0.01]                     # budget given to Redundancy; the rest goes to Learned


def matched_cut(scores, target_fpr):
    v = np.asarray(scores, float)
    rank = max(0, min(len(v) - 1, int(np.ceil((1 - target_fpr) * len(v))) - 1))
    cut = np.sort(v)[rank]
    above, tied = np.mean(v > cut), np.mean(v == cut)
    return cut, (np.clip((target_fpr - above) / tied, 0.0, 1.0) if tied else 0.0)


def prob(z, cut, tp):
    z = np.asarray(z, float)
    return (z > cut) + tp * (z == cut)


def load(tag):
    d = pd.read_csv(f'{tag}_base_scores.csv')
    d = d[d.target == 'BP'].copy().reset_index(drop=True)
    for name in BASE:
        d['z_' + name] = np.nan
    for case, g in d.groupby('case'):
        ctrl = g[g.k == 0].drop_duplicates('start_index'); n = len(ctrl); is_c = (g.k == 0).values
        for name, col in BASE.items():
            cs = np.sort(ctrl[col].values)
            cnt = np.maximum(n - np.searchsorted(cs, g[col].values, side='left') - is_c, 0)
            d.loc[g.index, 'z_' + name] = -np.log((cnt + 1) / (n + 1))
    d['z_Learned'] = d[['z_IF', 'z_AE', 'z_LSTM']].max(axis=1)
    d['z_Max evidence'] = d[['z_Redundancy', 'z_Learned']].max(axis=1)
    return d


def detections(d, label):
    rows = []
    for (mag, profile), c in d[d.k == 0].groupby(['mag', 'profile']):
        a = d[(d.mag == mag) & (d.profile == profile) & (d.k > 0)]
        groups = list(a.groupby(['case', 'k', 'devices']).groups.items())
        for det in ['Redundancy', 'Learned', 'LSTM', 'Max evidence']:
            cut, tp = matched_cut(c['z_' + det], TARGET_FPR)
            hit = pd.Series(prob(a['z_' + det], cut, tp), index=a.index)
            for (case, k, dv), idx in groups:
                rows.append(dict(set=label, mag=mag, profile=profile, detector=det, case=case, k=k, devices=dv, detection=hit.loc[idx].mean()))
        for ar in SPLITS:
            al = TARGET_FPR - ar
            (c1, t1), (c2, t2) = matched_cut(c.z_Redundancy, ar), matched_cut(c.z_Learned, al)
            miss = (1 - prob(a.z_Redundancy, c1, t1)) * (1 - prob(a.z_Learned, c2, t2))
            hit = pd.Series(1 - miss, index=a.index)
            name = f'OR {ar * 100:g}%+{al * 100:g}%'
            for (case, k, dv), idx in groups:
                rows.append(dict(set=label, mag=mag, profile=profile, detector=name, case=case, k=k, devices=dv, detection=hit.loc[idx].mean()))
    return rows


rows = []
for label, tag in SETS.items():
    rows += detections(load(tag), label)
det = pd.DataFrame(rows)
det.to_csv('pilot17_bp_fusion_detection.csv', index=False)
order = ['Redundancy', 'Learned', 'LSTM', 'Max evidence'] + [f'OR {a * 100:g}%+{(TARGET_FPR - a) * 100:g}%' for a in SPLITS]

pd.set_option('display.width', 220)
print('Arterial pressure: detection at a matched 10% false-alarm rate by number of compromised sources')
print(det.groupby(['set', 'k', 'detector']).detection.mean().unstack('detector')[order].round(2).to_string())
for label in SETS:
    s = det[det.set == label]
    kmax = int(s.k.max())
    for k in (1, kmax):
        print(f'\n{label}: detection by attack size (patient SD) with {k} compromised source(s)')
        print(s[s.k == k].groupby(['mag', 'detector']).detection.mean().unstack('detector')[order].round(2).to_string())

KEYS = ['set', 'detector', 'case', 'mag', 'profile']
ev = det[det.detection <= TARGET_FPR + MARGIN].sort_values(KEYS + ['k', 'detection'])
best = ev.groupby(KEYS).first().reset_index()[KEYS + ['k']].rename(columns={'k': 'cost'})
cells = det[KEYS].drop_duplicates().merge(best, how='left')
ad = cells.groupby(['set', 'detector', 'case', 'mag']).cost.min().reset_index()
print('\nArterial pressure: grid attack cost, attacker free to choose the profile (evasion found / mean cost)')
t = ad.groupby(['set', 'detector']).agg(evasion=('cost', lambda x: x.notna().mean()), cost=('cost', 'mean')).round(2).unstack('detector')
print(t.reindex(columns=pd.MultiIndex.from_product([['evasion', 'cost'], order])).to_string())
print('\nBy attack size (mean cost / evasion found), design cases:')
for label in SETS:
    a = ad[ad.set == label]
    print(f'-- {label}')
    print(a.groupby(['mag', 'detector']).cost.agg(['mean', lambda x: x.notna().mean()]).round(2).unstack('detector').to_string())
