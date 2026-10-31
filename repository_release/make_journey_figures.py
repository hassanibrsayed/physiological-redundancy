"""Figures for the whole journey of results, built only from the CSV files already in this folder.

Run:  .\\.venv\\Scripts\\python.exe -m pip install matplotlib      (once, if matplotlib is missing)
      .\\.venv\\Scripts\\python.exe make_journey_figures.py
Writes journey_figures/*.png and journey_figures/index.html (open the html in a browser).
A figure whose input files are missing is skipped with a message; nothing else is touched.

Figures
  01  Pilot 1: detection vs attack size, single-channel and coordinated attacks on heart rate
  02  Detection vs number of compromised devices (8 design cases and 12 replication cases)
  03  Detection vs attack size in patient SD, one compromised device, five detectors
  04  ROC curves (one compromised device vs clean windows), case-averaged
  05  Red-team attack cost vs attack size, with and without C4, 8 and 12 cases
  06  Evasion found vs attack size (share of attack goals the red team could hide)
  07  How the cost estimate moved through the pilots (grid, joint thresholds, red team, replication)
  08  Blood pressure: learned twin vs the other detectors, by attack size and device count
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

OUT = 'journey_figures'
os.makedirs(OUT, exist_ok=True)
TARGETS = ['HR', 'ETCO2', 'RR', 'BP']
TN = {'HR': 'Heart rate', 'ETCO2': 'EtCO2', 'RR': 'Respiratory rate', 'BP': 'Arterial MBP'}
COL = {'Redundancy': '#1f6fb4', 'Learned': '#d1495b', 'IF (all ch)': '#edae49', 'IF (8 ch)': '#f4a261',
       'Autoencoder': '#8c6bb1', 'LSTM': '#2a9d8f', 'Twin': '#2d6a4f', 'Combined (max)': '#6c757d',
       'Split OR (5%+5%)': '#6c757d', 'with C4': '#d1495b', 'without C4': '#1f6fb4'}
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 110, 'legend.frameon': False})
CAPTIONS = []


def need(*files):
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print('skipped (missing):', ', '.join(missing))
    return not missing


def save(fig, name, caption):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)
    CAPTIONS.append((name, caption))
    print('wrote', name)


def comb_det(prefix):
    return pd.read_csv(f'{prefix}_comb_detection.csv')


def red_team(prefix):
    c = pd.read_csv(f'{prefix}_cost.csv')
    return c[c['mode'] == 'timing-free']


def fig01():
    if not need('pilot_results.csv'):
        return
    d = pd.read_csv('pilot_results.csv')
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    titles = ['Attack on ECG heart rate only', 'Coordinated: HR and pleth pulse rate shifted together']
    for a, coord, title in zip(ax, [False, True], titles):
        s = d[d.coord == coord]
        for kind, ls in [('bias', '-'), ('drift', '--')]:
            g = s[s.kind == kind].groupby('mag')[['red', 'iso']].mean()
            a.plot(g.index, g.red, ls, marker='o', color=COL['Redundancy'], label=f'Redundancy ({kind})')
            a.plot(g.index, g.iso, ls, marker='s', color=COL['IF (8 ch)'], label=f'Isolation Forest ({kind})')
        a.set_title(title, fontsize=10); a.set_xlabel('Attack size (bpm)'); a.set_ylim(-0.02, 1.02)
    ax[0].set_ylabel('Fraction of attacks detected (1% false alarms)'); ax[0].legend(fontsize=8)
    save(fig, '01_pilot1_hr_attacks.png',
         'Pilot 1 (20 cases). Heart-rate vs pleth pulse-rate redundancy detects single-channel attacks that Isolation Forest '
         'misses; when the attacker shifts both channels together the redundancy detector falls to chance.')


def fig02():
    if not need('pilot12_comb_detection.csv', 'pilot14_comb_detection.csv'):
        return
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for a, t in zip(axes, TARGETS):
        for p, ls, lab in [('pilot12', '-', '8 design cases'), ('pilot14', '--', '12 new cases')]:
            d = comb_det(p); d = d[d.target == t]
            for det in ['Redundancy', 'Learned']:
                g = d[d.detector == det].groupby('k').detection.mean()
                a.plot(g.index, g.values, ls, marker='o', color=COL[det], label=f'{det}, {lab}')
        a.set_title(TN[t]); a.set_xlabel('Compromised devices'); a.set_xticks([1, 2, 3]); a.set_ylim(0, 1.02)
    axes[0].set_ylabel('Detection at matched 10% false alarms'); axes[0].legend(fontsize=7)
    save(fig, '02_detection_vs_devices.png',
         'Detection falls as the attacker falsifies more independent devices consistently. Learned detectors (IF, autoencoder, LSTM) '
         'stay low and flat. Solid: 8 design cases; dashed: 12 replication cases.')


def fig03():
    if not need('pilot12_comb_detection.csv'):
        return
    d = comb_det('pilot12'); d = d[d.k == 1]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for a, t in zip(axes, TARGETS):
        for det in ['Redundancy', 'IF (all ch)', 'Autoencoder', 'LSTM', 'Split OR (5%+5%)']:
            g = d[(d.target == t) & (d.detector == det)].groupby('mag').detection.mean()
            a.plot(g.index, g.values, marker='o', color=COL[det], label=det, ls='--' if det.startswith('Split') else '-')
        a.set_xscale('log'); a.set_xticks([0.25, 0.5, 1, 2]); a.set_xticklabels(['0.25', '0.5', '1', '2'])
        a.set_title(TN[t]); a.set_xlabel('Attack size (patient SD)'); a.set_ylim(0, 1.02)
    axes[0].set_ylabel('Detection at matched 10% false alarms'); axes[0].legend(fontsize=7)
    save(fig, '03_detection_vs_attack_size.png',
         'One compromised device (8 design cases, both signs, three onset profiles). Redundancy is far ahead for EtCO2, HR and RR; '
         'for blood pressure nothing detects small shifts.')


def fig04():
    if not need('pilot12_base_scores.csv'):
        return
    s_all = pd.read_csv('pilot12_base_scores.csv')
    cols = {'Redundancy': 'red_target_score', 'IF (8 ch)': 'iso_target_score', 'IF (all ch)': 'score_IF_all',
            'Autoencoder': 'score_AE', 'LSTM': 'score_LSTM'}
    grid = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for a, t in zip(axes, TARGETS):
        s = s_all[s_all.target == t]
        for name, col in cols.items():
            tprs, aucs = [], []
            for _, g in s.groupby('case'):
                c, at = g[g.k == 0], g[g.k == 1]
                if len(c) < 5 or len(at) < 5:
                    continue
                y = np.r_[np.zeros(len(c)), np.ones(len(at))]; sc = np.r_[c[col], at[col]]
                f, tp, _ = roc_curve(y, sc); tprs.append(np.interp(grid, f, tp)); aucs.append(roc_auc_score(y, sc))
            if tprs:
                a.plot(grid, np.mean(tprs, axis=0), color=COL[name], label=f'{name} (AUC {np.mean(aucs):.2f})')
        a.plot([0, 1], [0, 1], color='grey', lw=0.8, ls=':')
        a.set_title(TN[t]); a.set_xlabel('False-alarm rate'); a.set_xlim(0, 1); a.set_ylim(0, 1.02)
    axes[0].set_ylabel('Detection rate'); [ax.legend(fontsize=7, loc='lower right') for ax in axes]
    save(fig, '04_roc_curves.png',
         'ROC curves, one compromised device versus clean windows, averaged over cases (vertical averaging). AUC is the mean of per-case AUCs.')


def fig05_06():
    if not need('pilot13_cost.csv', 'pilot15_cost.csv'):
        return
    cost = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    evas = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for i, t in enumerate(TARGETS):
        for p, ls, lab in [('pilot13', '-', '8 cases'), ('pilot15', '--', '12 new cases')]:
            c = red_team(p); c = c[c.target == t]
            for det in ['with C4', 'without C4']:
                cc = c[c.detector == det]
                m = cc.groupby('mag').cost.mean()
                e = cc.groupby('mag').cost.apply(lambda x: x.notna().mean())
                cost[1][i].plot(m.index, m.values, ls, marker='o', color=COL[det], label=f'{det}, {lab}')
                evas[1][i].plot(e.index, e.values, ls, marker='o', color=COL[det], label=f'{det}, {lab}')
        for fig_, ax in (cost, evas):
            ax[i].set_xscale('log'); ax[i].set_xticks([0.5, 1, 2]); ax[i].set_xticklabels(['0.5', '1', '2'])
            ax[i].set_title(TN[t]); ax[i].set_xlabel('Attack goal (patient SD)')
    cost[1][0].set_ylabel('Devices the attacker must compromise'); cost[1][0].set_ylim(0.8, 3.4); cost[1][0].legend(fontsize=7)
    evas[1][0].set_ylabel('Share of goals the red team can hide'); evas[1][0].set_ylim(0, 1.05); evas[1][0].legend(fontsize=7)
    save(cost[0], '05_redteam_cost_vs_size.png',
         'Black-box red team, timing-free, joint thresholds. Mean number of devices in the cheapest evading set (over goals where one was found). '
         'Cost rises with the size of the deviation the attacker wants; RR needs about three devices, BP about one.')
    save(evas[0], '06_redteam_evasion_vs_size.png',
         'Share of attack goals for which the red team found an evading attack. With C4 the rate-of-change checks leave little room on flat integer RR channels.')


def fig07():
    files = ['pilot9_comb_cost.csv', 'pilot12_comb_cost.csv', 'pilot11_cost.csv', 'pilot13_cost.csv', 'pilot15_cost.csv']
    if not need(*files):
        return

    def grid(p):
        c = pd.read_csv(f'{p}_comb_cost.csv'); c = c[c.detector == 'Redundancy']
        return c.groupby(['case', 'target', 'mag']).cost.min().reset_index().groupby('target').cost.mean()

    def red(p, det):
        c = red_team(p); return c[c.detector == det].groupby('target').cost.mean()
    stages = [('Grid attack, Bonferroni thresholds (8 cases)', grid('pilot9'), '#bcd4e6'),
              ('Grid attack, joint thresholds (8)', grid('pilot12'), '#7fb0d6'),
              ('Red team, Bonferroni thresholds (8)', red('pilot11', 'with C4'), '#f2c6c6'),
              ('Red team, joint, with C4 (8)', red('pilot13', 'with C4'), '#e08a8a'),
              ('Red team, joint, no C4 (8)', red('pilot13', 'without C4'), '#1f6fb4'),
              ('Red team, joint, no C4 (12 new cases)', red('pilot15', 'without C4'), '#0b3d6b')]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.13; x = np.arange(len(TARGETS))
    for i, (lab, s, color) in enumerate(stages):
        v = [s.get(t, np.nan) for t in TARGETS]
        ax.bar(x + (i - 2.5) * w, v, w, label=lab, color=color)
    ax.set_xticks(x); ax.set_xticklabels([TN[t] for t in TARGETS]); ax.set_ylabel('Estimated attack cost (devices)')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.set_ylim(0, 3.8)
    save(fig, '07_cost_estimate_journey.png',
         'How the attack-cost estimate moved: the grid attacker overstated it, Bonferroni thresholds understated it, joint thresholds and the red team '
         'settled it, and the 12 replication cases agree. Missing bar = no evading set found. Means are over goals where one was found.')


def fig08():
    if not need('pilot10_bp_detection.csv'):
        return
    d = pd.read_csv('pilot10_bp_detection.csv')
    dets = ['Redundancy', 'Learned', 'Twin', 'Combined (max)']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    s = d[d.k == 1]
    for det in dets:
        g = s[s.detector == det].groupby('mag').detection.mean()
        ax[0].plot(g.index, g.values, marker='o', color=COL[det], label=det)
        g = d[d.detector == det].groupby('k').detection.mean()
        ax[1].plot(g.index, g.values, marker='o', color=COL[det], label=det)
    ax[0].set_xscale('log'); ax[0].set_xticks([0.25, 0.5, 1, 2]); ax[0].set_xticklabels(['0.25', '0.5', '1', '2'])
    ax[0].set_xlabel('Attack size (patient SD), arterial line only'); ax[0].set_ylabel('Detection at matched 10% false alarms')
    ax[1].set_xlabel('Compromised devices'); ax[1].set_xticks([1, 2, 3]); ax[0].legend(fontsize=8)
    for a in ax:
        a.set_ylim(0, 1.02)
    save(fig, '08_bp_twin.png',
         'Blood pressure. The learned twin is the best single-device detector but its gain comes from the EV1000 second arterial source; '
         'once EV1000 and the cuff are also compromised it sees nothing.')


for f in (fig01, fig02, fig03, fig04, fig05_06, fig07, fig08):
    try:
        f()
    except Exception as e:                      # keep going so one bad input does not block the rest
        print(f'{f.__name__} failed: {e}')

STEPS = [
    ('Pilot 1', 'HR vs pleth pulse rate, 20 cases: redundancy detects attacks Isolation Forest misses; coordinated attacks evade both.'),
    ('Pilots 2-5', 'Invariants R1-R5 and C1-C4 on 8 rich cases, detection against number of compromised devices, independent calibration, bootstrap over cases.'),
    ('Pilot 6', 'Arterial-waveform pulse rate (R2) raises the heart-rate cost from 2 to 3 devices.'),
    ('Pilot 7', 'Learned baselines (IF on all channels, autoencoder, LSTM): they ignore device count, and win only for large shifts.'),
    ('Pilot 8', 'Combined detector: naive max fusion dilutes redundancy; a split budget recovers part of it.'),
    ('Pilot 9', 'Attack size in patient SD, both signs: cost rises with size; BP is unprotected.'),
    ('Pilot 10', 'Learned BP twin: helps only where EV1000 gives a second arterial source.'),
    ('Pilot 11', 'Black-box red team: grid cost slightly overstated; found a threshold-calibration flaw.'),
    ('Joint thresholds', 'One joint cut instead of a Bonferroni split; RR channels are flat integers, so RR evidence is reported without C4.'),
    ('Replication', '12 unused cases: ordering BP < EtCO2 ~ HR < RR and magnitudes replicate.'),
]
with open(os.path.join(OUT, 'index.html'), 'w', encoding='utf-8') as fh:
    fh.write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>Physiological redundancy: journey of results</title>'
             '<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}'
             'img{width:100%;border:1px solid #ddd;border-radius:6px}figure{margin:24px 0}figcaption{font-size:14px;color:#444;margin-top:6px}'
             'li{margin:4px 0}h1{font-size:24px}</style></head><body>')
    fh.write('<h1>Physiological redundancy as a security invariant: journey of results</h1><ol>')
    for k, v in STEPS:
        fh.write(f'<li><b>{k}.</b> {v}</li>')
    fh.write('</ol>')
    for name, cap in CAPTIONS:
        fh.write(f'<figure><img src="{name}" alt="{name}"><figcaption>{cap}</figcaption></figure>')
    fh.write('</body></html>')
print('wrote journey_figures/index.html')
