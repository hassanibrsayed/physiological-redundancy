"""Draw the three full-width figures of manuscript v4 directly from the result files (no single-panel PNGs are reused).

Why a new script: the v4 figures built from make_journey_figures.py used the untuned learned detectors, so they disagreed with Table 1,
and the one-row strips were unreadable at column width. Here Figure 1 uses pilot12tuned_comb_detection.csv and
pilot14tuned_comb_detection.csv (tuned learned detectors, joint thresholds), averaged exactly as Table 1 (mean over cases, sizes and
profiles), and every figure is 7.2 inches wide with a 2 x 4 or 1 x 3 grid, linear axes with ticks only at the attack sizes, and
"sources" in the labels (devices only for the trust-domain bars).

Usage:  .\\.venv\\Scripts\\python.exe make_figures_v4.py
Input:  pilot12tuned_comb_detection.csv, pilot14tuned_comb_detection.csv, pilot13_cost.csv, pilot15_cost.csv,
        pilot10_bp_detection.csv, trust_domain_cost.csv
Output: journey_figures/fig_v4_detection.png, fig_v4_redteam.png, fig_v4_bp_devices.png"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'journey_figures'
os.makedirs(OUT, exist_ok=True)
TARGETS = ['BP', 'ETCO2', 'HR', 'RR']
TN = {'BP': 'Arterial MBP', 'ETCO2': 'EtCO2', 'HR': 'Heart rate', 'RR': 'Respiratory rate'}
COL = {'Redundancy': '#1f6fb4', 'Learned': '#d1495b', 'IF (all ch)': '#edae49', 'Autoencoder': '#8c6bb1', 'LSTM': '#2a9d8f',
       'Twin': '#2d6a4f', 'Combined (max)': '#6c757d', 'with C4': '#d1495b', 'without C4': '#1f6fb4'}
plt.rcParams.update({'font.size': 7.5, 'axes.grid': True, 'grid.alpha': 0.25, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 6.5, 'axes.titlesize': 8})
W = 7.2


def letter(ax, s):
    ax.text(-0.02, 1.12, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')


def size_axis(ax, ticks):
    """Equally spaced categorical ticks at the attack sizes, so that 0.25 and 0.5 never overlap."""
    ax.set_xticks(range(len(ticks))); ax.set_xticklabels([f'{t:g}' for t in ticks]); ax.minorticks_off()


def xs(index, ticks):
    return [ticks.index(round(float(v), 2)) for v in index]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=220)
    plt.close(fig)
    print('wrote', name)


def fig_detection():
    d12 = pd.read_csv('pilot12tuned_comb_detection.csv'); d14 = pd.read_csv('pilot14tuned_comb_detection.csv')
    fig, axes = plt.subplots(2, 4, figsize=(W, 4.3), sharey=True)
    for a, t in zip(axes[0], TARGETS):
        for d, ls, lab in [(d12, '-', 'design'), (d14, '--', 'replication')]:
            s = d[d.target == t]
            for det in ['Redundancy', 'Learned']:
                g = s[s.detector == det].groupby('k').detection.mean()
                a.plot(g.index, g.values, ls, marker='o', ms=3, color=COL[det], label=f'{det}, {lab}')
        a.set_title(TN[t]); a.set_xlabel('Compromised sources'); a.set_xticks([1, 2, 3]); a.set_ylim(0, 1.02)
    axes[0][0].set_ylabel('Detection at matched\n10% false alarms'); axes[0][0].legend(loc='upper right')
    d1 = d12[d12.k == 1]
    for a, t in zip(axes[1], TARGETS):
        for det in ['Redundancy', 'IF (all ch)', 'Autoencoder', 'LSTM']:
            g = d1[(d1.target == t) & (d1.detector == det)].groupby('mag').detection.mean()
            a.plot(xs(g.index, [0.25, 0.5, 1, 2]), g.values, marker='o', ms=3, color=COL[det], label=det)
        size_axis(a, [0.25, 0.5, 1, 2]); a.set_xlabel('Attack size (patient SD)'); a.set_ylim(0, 1.02)
    axes[1][0].set_ylabel('Detection at matched\n10% false alarms'); axes[1][0].legend(loc='upper left')
    letter(axes[0][0], '(a)'); letter(axes[1][0], '(b)')
    save(fig, 'fig_v4_detection.png')


def fig_redteam():
    fig, axes = plt.subplots(2, 4, figsize=(W, 4.3))
    for i, t in enumerate(TARGETS):
        for p, ls, lab in [('pilot13', '-', 'design'), ('pilot15', '--', 'replication')]:
            c = pd.read_csv(f'{p}_cost.csv'); c = c[(c['mode'] == 'timing-free') & (c.target == t)]
            for det in ['with C4', 'without C4']:
                cc = c[c.detector == det]
                m = cc.groupby('mag').cost.mean(); e = cc.groupby('mag').cost.apply(lambda x: x.notna().mean())
                axes[0][i].plot(xs(m.index, [0.5, 1, 2]), m.values, ls, marker='o', ms=3, color=COL[det], label=f'{det}, {lab}')
                axes[1][i].plot(xs(e.index, [0.5, 1, 2]), e.values, ls, marker='o', ms=3, color=COL[det], label=f'{det}, {lab}')
        for r in (0, 1):
            size_axis(axes[r][i], [0.5, 1, 2])
        axes[0][i].set_title(TN[t]); axes[0][i].set_ylim(0.8, 3.4); axes[1][i].set_ylim(0, 1.05)
        axes[1][i].set_xlabel('Attack goal (patient SD)')
    axes[0][0].set_ylabel('Sources the attacker\nmust compromise'); axes[0][0].legend(loc='upper left')
    axes[1][0].set_ylabel('Share of goals with\nan evading attack')
    letter(axes[0][0], '(a)'); letter(axes[1][0], '(b)')
    save(fig, 'fig_v4_redteam.png')


def fig_bp_devices():
    d = pd.read_csv('pilot10_bp_detection.csv')
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.6), gridspec_kw={'width_ratios': [1, 1, 1.15]})
    for det in ['Redundancy', 'Learned', 'Twin', 'Combined (max)']:
        g = d[(d.k == 1) & (d.detector == det)].groupby('mag').detection.mean()
        ax[0].plot(xs(g.index, [0.25, 0.5, 1, 2]), g.values, marker='o', ms=3, color=COL[det], label=det)
        g = d[d.detector == det].groupby('k').detection.mean()
        ax[1].plot(g.index, g.values, marker='o', ms=3, color=COL[det], label=det)
    size_axis(ax[0], [0.25, 0.5, 1, 2]); ax[0].set_xlabel('Attack size (patient SD), arterial line only')
    ax[0].set_ylabel('Detection at matched\n10% false alarms'); ax[0].legend(loc='upper left')
    ax[1].set_xlabel('Compromised sources'); ax[1].set_xticks([1, 2, 3])
    for a in ax[:2]:
        a.set_ylim(0, 1.02)
    t = pd.read_csv('trust_domain_cost.csv'); t = t[(t.detector == 'without C4') & (t['mode'] == 'timing-free')]
    m = t.groupby('target').agg(sources=('sources', 'mean'), devices=('domains_A', 'mean')).reindex(['HR', 'BP', 'ETCO2', 'RR'])
    x = range(len(m)); w = 0.38
    ax[2].bar([i - w / 2 for i in x], m.sources, w, color='#1f6fb4', label='Sources')
    ax[2].bar([i + w / 2 for i in x], m.devices, w, color='#edae49', label='Devices')
    ax[2].set_xticks(list(x)); ax[2].set_xticklabels([TN[k] for k in m.index], rotation=20, ha='right')
    ax[2].set_ylabel('Cost (no C4)'); ax[2].legend(loc='upper right'); ax[2].grid(axis='x', alpha=0)
    letter(ax[0], '(a)'); letter(ax[1], '(b)'); letter(ax[2], '(c)')
    save(fig, 'fig_v4_bp_devices.png')


for f in (fig_detection, fig_redteam, fig_bp_devices):
    try:
        f()
    except Exception as e:
        print(f'{f.__name__} failed: {e}')
