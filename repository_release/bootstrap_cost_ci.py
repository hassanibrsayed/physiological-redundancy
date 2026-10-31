"""Case-level bootstrap confidence intervals for the red-team attack cost (Table 1 of the manuscript).

No new experiment is run: the script resamples whole cases (with replacement, separately inside the 8 design cases and the 12
replication cases) from pilot13_cost.csv and pilot15_cost.csv, recomputes the mean cost over the goals of the resampled cases, and
reports the 2.5th and 97.5th percentiles over 2,000 resamples. Goals without an evading attack have no cost and drop out of the mean, as in
the table, so the share of goals with an evasion gets its own interval. It also bootstraps the differences between targets that carry the claim
"the ordering replicated", and the difference between the design and the replication cost for each target.

Usage:  .\\.venv\\Scripts\\python.exe bootstrap_cost_ci.py
Input:  pilot13_cost.csv (design, joint thresholds), pilot15_cost.csv (replication), timing-free rows only
Output: bootstrap_cost_ci.csv and printed tables"""
import numpy as np, pandas as pd

B = 2000
SEED = 20261027
rng = np.random.default_rng(SEED)
SETS = {'design': 'pilot13_cost.csv', 'replication': 'pilot15_cost.csv'}
TARGETS = ['BP', 'ETCO2', 'HR', 'RR']
DETS = ['without C4', 'with C4']

data = {}
for name, path in SETS.items():
    d = pd.read_csv(path)
    d = d[d['mode'] == 'timing-free']
    data[name] = {c: g for c, g in d.groupby('case')}          # case -> its goals
print({k: len(v) for k, v in data.items()}, 'cases per set')


def stat(frames, target, det):
    """mean cost over goals with an evasion, and share of goals with an evasion"""
    g = pd.concat(frames)
    g = g[(g.target == target) & (g.detector == det)]
    return g.cost.mean(), g.cost.notna().mean()


def draw(name):
    cases = list(data[name])
    pick = rng.choice(cases, size=len(cases), replace=True)
    return [data[name][c] for c in pick]


boot = {k: [] for k in ('design', 'replication', 'pooled')}
for _ in range(B):
    bd, br = draw('design'), draw('replication')
    row = {}
    for label, frames in (('design', bd), ('replication', br), ('pooled', bd + br)):
        row[label] = {(t, det): stat(frames, t, det) for t in TARGETS for det in DETS}
    for k in boot:
        boot[k].append(row[k])

full = {'design': list(data['design'].values()), 'replication': list(data['replication'].values())}
full['pooled'] = full['design'] + full['replication']


def ci(vals):
    v = np.array([x for x in vals if not np.isnan(x)])
    return np.percentile(v, 2.5), np.percentile(v, 97.5), len(v)


rows = []
for label in ('design', 'replication', 'pooled'):
    for det in DETS:
        for t in TARGETS:
            mean, share = stat(full[label], t, det)
            lo, hi, n = ci([b[(t, det)][0] for b in boot[label]])
            slo, shi, _ = ci([b[(t, det)][1] for b in boot[label]])
            rows.append(dict(set=label, detector=det, target=t, mean_cost=round(mean, 2), ci_low=round(lo, 2), ci_high=round(hi, 2),
                             evasion_share=round(share, 2), evasion_ci_low=round(slo, 2), evasion_ci_high=round(shi, 2), valid_resamples=n))
out = pd.DataFrame(rows)
out.to_csv('bootstrap_cost_ci.csv', index=False)
pd.set_option('display.width', 200)
print(f'\nRed-team cost, case-level bootstrap, {B} resamples, 95% percentile intervals (seed {SEED})')
print(out.to_string(index=False))

print('\nDifferences between targets (pooled over the 20 cases, without C4): mean and 95% interval; share of resamples with a positive difference')
pairs = [('HR', 'ETCO2'), ('ETCO2', 'BP'), ('RR', 'HR')]
for a, b in pairs:
    for det in DETS:
        diffs = [bb[(a, det)][0] - bb[(b, det)][0] for bb in boot['pooled']]
        pa, _ = stat(full['pooled'], a, det); pb, _ = stat(full['pooled'], b, det)
        lo, hi, n = ci(diffs)
        pos = np.mean([d > 0 for d in diffs if not np.isnan(d)])
        print(f'{a} - {b}, {det}: {pa - pb:+.2f} [{lo:+.2f}, {hi:+.2f}]  P(diff > 0) = {pos:.2f}  ({n} valid resamples)')

print('\nDifference replication - design (without C4): mean and 95% interval')
for t in TARGETS:
    diffs = [r[(t, 'without C4')][0] - d[(t, 'without C4')][0] for r, d in zip(boot['replication'], boot['design'])]
    md, _ = stat(full['design'], t, 'without C4'); mr, _ = stat(full['replication'], t, 'without C4')
    lo, hi, n = ci(diffs)
    print(f'{t}: {mr - md:+.2f} [{lo:+.2f}, {hi:+.2f}]')
