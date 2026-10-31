"""Pilot 11: black-box red team against the deployed redundancy detector.

The attacker queries the detector (alarm / score on a 60 s window) and searches for the cheapest evasive attack.
Goal for each (case, target, size, sign): displace the target channel by sign * size * SD_patient (final value),
with the average displacement over the window at least half of that. For every device subset S that contains the
target device (|S| = cost) the attacker optimises, within a budget of 60 queries (30 random, then 30 mutations of
the best three):
  * ramp length of the target device (1-60 s),
  * for every other compromised device an offset factor (0.5-1.5 of the target offset, i.e. not necessarily the
    same shift) and an onset lag (0-10 s),
  * in 'timing-free' mode, the start time among 40 candidate windows of the test segment (the attacker knows
    when the cuff will not measure, and when the patient is quiet); in 'fixed-start' mode the start is one
    random window.
The attack must keep every shifted channel inside its broad physiological range. It succeeds when no invariant
of the detector exceeds its threshold anywhere in the window. Detector: the target-specific redundancy detector
of Pilot 5/6/9 with thresholds calibrated on the calibration segment (10% window FPR), with and without the C4
rate constraints. Cost = smallest |S| for which the search succeeds.
This is a bounded search, so the cost it finds is an upper bound on the true minimum cost.
Usage:  .\\.venv\\Scripts\\python.exe pilot11_redteam.py       Outputs: pilot11_redteam.csv, pilot11_cost.csv"""
import glob, itertools, os, time, warnings
import numpy as np, pandas as pd
import pilot6 as p6
warnings.filterwarnings('ignore')

PRE, L = p6.PRE, p6.L
JOINT = os.environ.get('JOINT') == '1'          # JOINT=1: joint thresholds instead of the Bonferroni split
TAG = os.environ.get('TAG', 'pilot11')          # output prefix, e.g. TAG=pilot13
MAGS = [0.5, 1.0, 2.0]                 # attack size in patient SD
Q1, Q2 = 30, 30                        # queries: random, then mutations
N_START = 40
rng = np.random.default_rng(11)


def calibrate(D):
    T = len(D['HR']); train_end, cal_end = int(T * 0.30), int(T * 0.70)
    ok = {k: np.isfinite(D[k][:train_end]).mean() > 0.05 for k in D}
    par = {}
    for name, (fn, need, kind, floor) in p6.INV.items():
        if not all(ok[c] for c in need):
            continue
        res = fn(D)[:train_end]; res = res[np.isfinite(res)]
        if kind == 'hinge':
            par[name] = None
        elif len(res) < 30:
            continue
        elif kind == 'rate':
            par[name] = (0.0, max(float(np.quantile(np.abs(res), 0.995)), floor))
        else:
            med = np.median(res); par[name] = (med, max(1.4826 * np.median(np.abs(res - med)), floor))
    cal_starts = list(range(train_end, cal_end - L + 1, L))
    maxima = {}
    for name in par:
        residual = p6.INV[name][0](D)
        sc_name = p6.inv_scores(D, {name: par[name]})[name]
        vals = np.array([np.nanmax(sc_name[st:st + L]) if np.isfinite(residual[st:st + L]).any() else np.nan
                         for st in cal_starts])             # aligned across invariants, NaN = not evaluated
        if np.isfinite(vals).sum() >= 10:
            maxima[name] = vals
    par = {n: par[n] for n in maxima}
    return par, maxima, ok, train_end, cal_end


def joint_thresholds(maxima, names, fpr):
    """Threshold of invariant i is tau * c_i (c_i = its own 90th percentile of clean window maxima, floor 0.5);
    tau makes the share of clean calibration windows with any alarm equal to fpr (see pilot9.py)."""
    c = {n: max(float(np.nanquantile(maxima[n], 0.90)), 0.5) for n in names}
    M = np.vstack([np.asarray(maxima[n], float) / c[n] for n in names])
    stat = np.nanmax(M, axis=0)
    tau = float(np.quantile(stat[np.isfinite(stat)], 1 - fpr))
    return {n: tau * c[n] for n in names}


def thresholds(par, maxima, target, with_c4):
    inv = [n for n in par if n in p6.TARGET_INVS[target] and (with_c4 or not n.startswith('C4_'))]
    if not inv:
        return None
    if JOINT:
        return joint_thresholds(maxima, inv, p6.TARGET_FPR)
    q = 1 - p6.TARGET_FPR / len(inv)
    return {n: np.nanquantile(maxima[n], q) for n in inv}


def excess(D, par, thr, s, shifts):
    """Largest normalised exceedance over the detector's thresholds; <= 0 means no alarm."""
    sl = slice(s - PRE, s + L)
    Dseg = {k: v[sl].copy() for k, v in D.items()}
    for c, dl in shifts.items():
        Dseg[c][PRE:] += dl
    sc = p6.inv_scores(Dseg, {n: par[n] for n in thr})
    return max((np.max(sc[n][PRE:]) - thr[n]) / max(abs(thr[n]), 1.0) for n in thr)


def build(devs, sub, params, off):
    """params: dict(rho, eps[i], lag[i]) -> channel shifts. Target device (index 0) has factor 1 and no lag."""
    t = np.arange(1, L + 1, dtype=float)
    shifts = {}
    for pos, i in enumerate(sub):
        f = 1.0 if pos == 0 else params['eps'][pos]
        lag = 0.0 if pos == 0 else params['lag'][pos]
        d = off * f * np.clip((t - lag) / params['rho'], 0, 1)
        for c in devs[i][1]:
            shifts[c] = d
    return shifts


def in_range(D, s, shifts):
    for c, d in shifts.items():
        v = D[c][s:s + L] + d; fin = np.isfinite(v); lo, hi = p6.RANGE[c]
        if fin.any() and ((v[fin] < lo).any() or (v[fin] > hi).any()):
            return False
    return True


def search(D, par, thr, devs, sub, off, starts, free_start):
    m = len(sub)

    def draw(base=None, sigma=0.15):
        if base is None:
            p = dict(rho=rng.uniform(1, L), eps=np.r_[1.0, rng.uniform(0.5, 1.5, m - 1)], lag=np.r_[0.0, rng.uniform(0, 10, m - 1)],
                     s=starts[rng.integers(len(starts))] if free_start else starts[0])
        else:
            p = dict(rho=float(np.clip(base['rho'] + sigma * L * rng.normal(), 1, L)),
                     eps=np.r_[1.0, np.clip(base['eps'][1:] + sigma * rng.normal(size=m - 1), 0.5, 1.5)],
                     lag=np.r_[0.0, np.clip(base['lag'][1:] + sigma * 10 * rng.normal(size=m - 1), 0, 10)],
                     s=(starts[rng.integers(len(starts))] if free_start and rng.random() < 0.3 else base['s']))
        return p

    def score(p):
        sh = build(devs, sub, p, off)
        if not in_range(D, p['s'], sh):
            return np.inf
        return excess(D, par, thr, p['s'], sh)

    pool = []
    for i in range(Q1):
        p = draw()
        if i < 3:       # seed the search with the attacker's obvious first guess: identical offset, no lag
            p['eps'] = np.ones(m); p['lag'] = np.zeros(m); p['rho'] = (10.0, 30.0, 60.0)[i]
        e = score(p); pool.append((e, p))
        if e <= 0:
            return e, p
    for _ in range(Q2):
        pool.sort(key=lambda x: x[0]); top = pool[:3]
        p = draw(base=top[rng.integers(len(top))][1]); e = score(p); pool.append((e, p))
        if e <= 0:
            return e, p
    return min(pool, key=lambda x: x[0])[0], min(pool, key=lambda x: x[0])[1]


def run_case(path):
    case = int(os.path.basename(path)[4:-4])
    D = p6.load(path); T = len(D['HR'])
    par, maxima, ok, train_end, cal_end = calibrate(D)
    rows = []
    for tname, tg in p6.TARGETS.items():
        prim = tg['primary']
        if not ok[prim]:
            continue
        devs = [d for i, d in enumerate(tg['devs']) if i == 0 or all(ok[c] for c in d[1])]
        sd = float(np.nanstd(D[prim][:train_end]))
        okp = np.isfinite(D[prim])
        cand = [s for s in range(cal_end + PRE, T - L - 1) if okp[s:s + L].all()]
        if len(cand) < N_START:
            continue
        starts = [int(x) for x in rng.choice(cand, N_START, replace=False)]
        subsets = [(0,) + c for r in range(len(devs)) for c in itertools.combinations(range(1, len(devs)), r)]
        for dname, with_c4 in [('with C4', True), ('without C4', False)]:
            thr = thresholds(par, maxima, tname, with_c4)
            if thr is None:
                continue
            for mag in MAGS:
                for sign in (1, -1):
                    off = sign * mag * sd
                    for mode, free in [('timing-free', True), ('fixed-start', False)]:
                        st = starts if free else [starts[int(rng.integers(len(starts)))]]
                        for sub in subsets:
                            e, p = search(D, par, thr, devs, sub, off, st, free)
                            rows.append(dict(case=case, target=tname, detector=dname, mag=mag, sign=sign, mode=mode,
                                             k=len(sub), devices='+'.join(devs[i][0] for i in sub), kmax=len(devs),
                                             best_excess=e, success=bool(e <= 0), start=p['s'], rho=p['rho'],
                                             eps=','.join(f'{x:.2f}' for x in p['eps'][1:]),
                                             lag=','.join(f'{x:.1f}' for x in p['lag'][1:]),
                                             cuff_in_window=bool(np.isfinite(D['NMBP'][p['s']:p['s'] + L]).any())))
    return rows


if __name__ == '__main__':
    allrows = []
    for f in sorted(glob.glob(os.path.join(p6.DATA, 'case*.csv'))):
        if f.endswith('_wave.csv'):
            continue
        t0 = time.time(); r = run_case(f); allrows += r
        print(f'{os.path.basename(f)}: {len(r)} searches, {time.time() - t0:.0f}s', flush=True)
    d = pd.DataFrame(allrows); d.to_csv(f'{TAG}_redteam.csv', index=False)
    KEYS = ['case', 'target', 'detector', 'mag', 'sign', 'mode']
    cost = (d[d.success].groupby(KEYS).k.min().rename('cost').reset_index())
    cells = d[KEYS + ['kmax']].drop_duplicates().merge(cost, how='left')
    cells.to_csv(f'{TAG}_cost.csv', index=False)
    pd.set_option('display.width', 200)
    print('\nRed-team cost (smallest compromised-device set found; evasion found = share of attack goals with a success):')
    t = cells.groupby(['detector', 'mode', 'target']).agg(goals=('cost', 'size'), evasion_found=('cost', lambda x: x.notna().mean()),
                                                           mean_cost=('cost', 'mean')).round(2)
    print(t.unstack('mode').to_string())
    print('\nBy attack size (with C4 / timing-free):')
    print(cells[(cells.detector == 'with C4') & (cells['mode'] == 'timing-free')].groupby(['target', 'mag']).cost
          .agg(['mean', lambda x: x.notna().mean()]).round(2).unstack('target').to_string())
    b = d[(d.target == 'BP') & (d.detector == 'with C4') & d.success]
    print('\nBP successful attacks: share whose window contains a cuff reading, by mode and k:')
    print(b.groupby(['mode', 'k']).cuff_in_window.agg(['size', 'mean']).round(2).to_string())
