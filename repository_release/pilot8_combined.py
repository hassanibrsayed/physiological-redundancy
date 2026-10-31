"""Pilot 8: combined detector (redundancy + learned) and the attack cost against it.

Reads pilot7_scores.csv, which holds, for every Pilot 6 attack window and no-attack control window, the window
score of the redundancy detector and of three learned detectors (Isolation Forest on all channels, autoencoder,
LSTM forecaster).

Scores live on different scales, so each detector's window score is turned into an evidence value
z = -log(p), where p is the (leave-one-out) fraction of that case's clean control windows scoring at least as
high. Fusion is the maximum z, i.e. the smallest p-value (an OR rule whose false-alarm rate is then fixed by
the matched cut below, not by a union bound):
  Redundancy        z of the redundancy detector
  Learned           max z over IF (all channels), autoencoder, LSTM
  Combined          max z over redundancy and the three learned detectors
Thresholds are the matched-FPR cut of Pilot 5: for each (target, magnitude, profile) the clean control windows
of all cases set a 10% false-alarm cut (ties randomised). This reuses the evaluation controls, so it is a
descriptive ROC-style comparison, not a deployable threshold and not an independent test.
Attack cost: smallest device set whose detection is at most the control FPR + 10 points (here 20%), per case,
magnitude and profile; the adaptive attacker takes the cheapest set over the three profiles.

Usage:  .\\.venv\\Scripts\\python.exe pilot8_combined.py     Outputs: pilot8_detection.csv, pilot8_cost.csv"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

TAG = sys.argv[1] if len(sys.argv) > 1 else None      # e.g. pilot9: read pilot9_base_scores.csv, write pilot9_comb_*.csv
INP = f'{TAG}_base_scores.csv' if TAG else 'pilot7_scores.csv'
OUTP = f'{TAG}_comb' if TAG else 'pilot8'
TARGET_FPR, MARGIN = 0.10, 0.10
BASE = {'Redundancy': 'red_target_score', 'IF (all ch)': 'score_IF_all', 'Autoencoder': 'score_AE', 'LSTM': 'score_LSTM'}
LEARNED = ['IF (all ch)', 'Autoencoder', 'LSTM']


def matched_cut(scores, target_fpr):
    v = np.asarray(scores, float)
    rank = max(0, min(len(v) - 1, int(np.ceil((1 - target_fpr) * len(v))) - 1))
    cut = np.sort(v)[rank]
    above, tied = np.mean(v > cut), np.mean(v == cut)
    tp = np.clip((target_fpr - above) / tied, 0.0, 1.0) if tied else 0.0
    return cut, tp


def detect(v, cut, tp):
    v = np.asarray(v, float)
    return np.mean(v > cut) + tp * np.mean(v == cut)


d = pd.read_csv(INP)
for name in BASE:
    d['z_' + name] = np.nan
for (case, target), g in d.groupby(['case', 'target']):
    ctrl = g[g.k == 0].drop_duplicates('start_index')
    n = len(ctrl)
    is_ctrl = (g.k == 0).values
    for name, col in BASE.items():
        cs = np.sort(ctrl[col].values)
        cnt = np.maximum(n - np.searchsorted(cs, g[col].values, side='left') - is_ctrl, 0)   # other controls scoring >= this window
        d.loc[g.index, 'z_' + name] = -np.log((cnt + 1) / (n + 1))
d['z_Learned'] = d[['z_' + x for x in LEARNED]].max(axis=1)
d['z_Combined'] = d[['z_Redundancy', 'z_Learned']].max(axis=1)
DETS = ['Redundancy', 'Learned', 'Combined'] + LEARNED

rows = []
for (target, mag, profile), c in d[d.k == 0].groupby(['target', 'mag', 'profile']):
    a = d[(d.target == target) & (d.mag == mag) & (d.profile == profile) & (d.k > 0)]
    for det in DETS:
        cut, tp = matched_cut(c['z_' + det], TARGET_FPR)
        for (case, k, devices), ak in a.groupby(['case', 'k', 'devices']):
            rows.append(dict(target=target, mag=mag, profile=profile, detector=det, case=case, k=k,
                             devices=devices, detection=detect(ak['z_' + det], cut, tp)))
# split-budget OR: alarm if the redundancy evidence passes its own cut (5% FPR) or the learned evidence passes a
# cut of equal budget (5% FPR each). Evidence values are heavily tied (about 30 control windows per case, so p has
# a resolution of 1/31); ties are randomised independently for the two detectors, as in matched_cut, which keeps
# the expected combined control FPR at or below 10%.
SPLIT = 'Split OR (5%+5%)'
fprs = []


def alarm_prob(z, cut, tp):
    z = np.asarray(z, float)
    return (z > cut) + tp * (z == cut)


for (target, mag, profile), c in d[d.k == 0].groupby(['target', 'mag', 'profile']):
    a = d[(d.target == target) & (d.mag == mag) & (d.profile == profile) & (d.k > 0)]
    (c1, t1), (c2, t2) = matched_cut(c.z_Redundancy, 0.05), matched_cut(c.z_Learned, 0.05)
    either = lambda x: 1 - (1 - alarm_prob(x.z_Redundancy, c1, t1)) * (1 - alarm_prob(x.z_Learned, c2, t2))
    fprs.append(either(c).mean())
    hit = pd.Series(either(a), index=a.index)
    for (case, k, devices), idx in a.groupby(['case', 'k', 'devices']).groups.items():
        rows.append(dict(target=target, mag=mag, profile=profile, detector=SPLIT, case=case, k=k,
                         devices=devices, detection=hit.loc[idx].mean()))
DETS.append(SPLIT)
print(f'Split OR realised control FPR (mean over target/magnitude/profile cells): {np.mean(fprs):.3f}\n')
det = pd.DataFrame(rows); det.to_csv(f'{OUTP}_detection.csv', index=False)

# attack cost against each detector
KEYS = ['detector', 'case', 'target', 'mag', 'profile']
ev = det[det.detection <= TARGET_FPR + MARGIN].sort_values(KEYS + ['k', 'detection'])
best = ev.groupby(KEYS).first().reset_index()[KEYS + ['k', 'devices']].rename(columns={'k': 'cost', 'devices': 'cheapest_set'})
cells = det[KEYS].drop_duplicates().merge(best, how='left')
cells.to_csv(f'{OUTP}_cost.csv', index=False)
adaptive = cells.groupby(['detector', 'case', 'target', 'mag']).cost.min().reset_index()

pd.set_option('display.width', 180)
print('Detection at matched 10% FPR by number of compromised devices (mean over cases, magnitudes, profiles):')
print(det.groupby(['target', 'k', 'detector']).detection.mean().unstack('detector')[DETS].round(2).to_string())
print('\nAttack cost vs each detector, attacker free to choose the profile (cheapest evading set; evasion = detection <= 20%):')
t = adaptive.groupby(['target', 'detector']).agg(cells=('cost', 'size'), evasion_found=('cost', lambda x: x.notna().mean()),
                                                  mean_cost=('cost', 'mean')).round(2).reset_index()
print(t.pivot(index='target', columns='detector', values=['evasion_found', 'mean_cost'])
        .reindex(columns=pd.MultiIndex.from_product([['evasion_found', 'mean_cost'], DETS])).to_string())
print('\nSame, by profile (evasion found / mean cost):')
p = cells.groupby(['target', 'profile', 'detector']).agg(evasion_found=('cost', lambda x: x.notna().mean()), mean_cost=('cost', 'mean')).round(2)
print(p.unstack('detector').reindex(columns=pd.MultiIndex.from_product([['evasion_found', 'mean_cost'], DETS])).to_string())
