"""Pilot 10: a learned per-patient twin for arterial blood pressure (the target no detector protects).

Twin: predicts how far arterial MBP has moved from its own recent baseline from how its independent partners
have moved from theirs. For time t, baseline(x) = median of x over [t-419 s, t-120 s] (causal, and always clean
for a 60 s attack because the attack is shorter than the 120 s gap). Features: EV1000 MBP - baseline(MBP),
cuff MBP - baseline(MBP) (both with presence flags, 0 when absent), and HR, SpO2, EtCO2 minus their own
baselines. Target: MBP - baseline(MBP). Model: gradient-boosted trees trained on the first 30% of each case
(needs no labels). Anomaly score: EWMA (0.1) of |residual| / robust residual scale from the calibration
segment; a window's score is its maximum over the 60 attacked seconds.
Attacks follow Pilot 9 exactly: a device in the compromised set shifts its channels by the same offset, so
compromising EV1000 and the cuff as well as the arterial line makes the partner features agree with the
shifted MBP and the twin sees no residual.

Evaluation reuses the Pilot 9 BP attack windows and the matched-10%-FPR cut of Pilot 8 (controls of all cases;
evidence z = -log p from each case's own clean control windows). Descriptive, not an independent test.
Usage:  .\\.venv\\Scripts\\python.exe pilot10_bp_twin.py     (needs pilot9_base_scores.csv)
Outputs: pilot10_bp_scores.csv, pilot10_bp_detection.csv, pilot10_bp_cost.csv, pilot10_bp_diagnostics.csv"""
import glob, os, warnings
import numpy as np, pandas as pd
from scipy.signal import lfilter
from sklearn.ensemble import HistGradientBoostingRegressor
import pilot6 as p6
warnings.filterwarnings('ignore')

PRE, L, A = p6.PRE, p6.L, p6.A
GAP, BASE_N = 120, 300          # baseline = median over the BASE_N seconds that end GAP seconds before t
TARGET_FPR, MARGIN = 0.10, 0.10
DEVS = {d[0]: d[1] for d in p6.TARGETS['BP']['devs']}
DEV_COL = {'EVM': 0, 'NMBP': 2}   # feature columns that shift when that device is compromised


def ew(x):
    return lfilter([A], [1, -(1 - A)], x, axis=-1)


def baseline(x):
    return pd.Series(x).rolling(BASE_N, min_periods=60).median().shift(GAP).values


def features(D):
    B = baseline(D['MBP'])
    cols = []
    for ch in ['EVM', 'NMBP']:
        m = np.isfinite(D[ch]); cols += [np.where(m, D[ch] - B, 0.0), m.astype(float)]
    for ch in ['HR', 'SPO2', 'ET']:
        d = D[ch] - baseline(D[ch]); cols.append(np.where(np.isfinite(d), d, 0.0))
    X = np.column_stack(cols)       # [dEVM, mEVM, dNMBP, mNMBP, dHR, dSPO2, dET]
    return X, D['MBP'] - B, B


def run_case(path, rows):
    case = int(os.path.basename(path)[4:-4])
    sub = rows[rows.case == case].copy()
    D = p6.load(path); T = len(D['HR'])
    train_end, cal_end = int(T * 0.30), int(T * 0.70)
    X, y, B = features(D)
    ok = np.isfinite(y) & np.isfinite(B)
    tr = np.where(ok[:train_end])[0]
    if len(tr) < 500:
        return None, None
    twin = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=4, random_state=0).fit(X[tr], y[tr])
    cal = np.where(ok[train_end:cal_end])[0] + train_end
    res_cal = y[cal] - twin.predict(X[cal])
    scale = max(1.4826 * np.median(np.abs(res_cal - np.median(res_cal))), 1.0)
    r2 = 1 - np.var(res_cal) / max(np.var(y[cal]), 1e-9)
    diag = dict(case=case, train_rows=len(tr), cal_r2=r2, resid_scale=scale, has_evm=bool(X[:, 1].mean() > 0.05),
                has_cuff=bool(X[:, 3].mean() > 0.01))
    n = len(sub)
    Y = np.empty((n, PRE + L)); XS = np.empty((n, PRE + L, X.shape[1]))
    for j, (_, r) in enumerate(sub.iterrows()):
        s = int(r.start_index); sl = slice(s - PRE, s + L)
        Y[j], XS[j] = y[sl], X[sl]
        if r.k > 0:
            delta = r.offset * p6.PROFILES[r.profile]
            for dev in r.devices.split('+'):
                if dev == 'ART':
                    Y[j, PRE:] += delta
                elif dev in ('NIBP', 'EV1000'):
                    c = DEV_COL['NMBP' if dev == 'NIBP' else 'EVM']
                    XS[j, PRE:, c] += delta * XS[j, PRE:, c + 1]       # only where the partner reports
    pred = twin.predict(XS.reshape(-1, X.shape[1])).reshape(n, PRE + L)
    sub['score_Twin'] = ew(np.abs(Y - pred) / scale)[:, -L:].max(1)
    return sub, diag


def matched_cut(scores, target_fpr):
    v = np.asarray(scores, float)
    rank = max(0, min(len(v) - 1, int(np.ceil((1 - target_fpr) * len(v))) - 1))
    cut = np.sort(v)[rank]
    above, tied = np.mean(v > cut), np.mean(v == cut)
    return cut, (np.clip((target_fpr - above) / tied, 0.0, 1.0) if tied else 0.0)


def prob(z, cut, tp):
    z = np.asarray(z, float)
    return (z > cut) + tp * (z == cut)


def main():
    base = pd.read_csv('pilot9_base_scores.csv')
    bp = base[base.target == 'BP'].copy()
    parts, diags = [], []
    for f in sorted(glob.glob(os.path.join(p6.DATA, 'case*.csv'))):
        if f.endswith('_wave.csv'):
            continue
        out, dg = run_case(f, bp)
        if out is not None:
            parts.append(out); diags.append(dg); print(dg, flush=True)
    d = pd.concat(parts); d.to_csv('pilot10_bp_scores.csv', index=False)
    pd.DataFrame(diags).to_csv('pilot10_bp_diagnostics.csv', index=False)

    cols = {'Redundancy': 'red_target_score', 'IF (all ch)': 'score_IF_all', 'Autoencoder': 'score_AE',
            'LSTM': 'score_LSTM', 'Twin': 'score_Twin'}
    for name in cols:
        d['z_' + name] = np.nan
    for case, g in d.groupby('case'):
        ctrl = g[g.k == 0].drop_duplicates('start_index'); n = len(ctrl); is_c = (g.k == 0).values
        for name, col in cols.items():
            cs = np.sort(ctrl[col].values)
            cnt = np.maximum(n - np.searchsorted(cs, g[col].values, side='left') - is_c, 0)
            d.loc[g.index, 'z_' + name] = -np.log((cnt + 1) / (n + 1))
    d['z_Learned'] = d[['z_IF (all ch)', 'z_Autoencoder', 'z_LSTM']].max(axis=1)
    d['z_Combined (max)'] = d[['z_Redundancy', 'z_Learned', 'z_Twin']].max(axis=1)
    single = ['Redundancy', 'Learned', 'Twin', 'Combined (max)']
    ORS = {'Redundancy OR Twin (5%+5%)': ['z_Redundancy', 'z_Twin'],
           'Redundancy OR Twin OR Learned (3.3% each)': ['z_Redundancy', 'z_Twin', 'z_Learned']}
    rows = []
    for (mag, profile), c in d[d.k == 0].groupby(['mag', 'profile']):
        a = d[(d.mag == mag) & (d.profile == profile) & (d.k > 0)]
        for det in single:
            cut, tp = matched_cut(c['z_' + det], TARGET_FPR)
            hit = pd.Series(prob(a['z_' + det], cut, tp), index=a.index)
            for (case, k, dv), idx in a.groupby(['case', 'k', 'devices']).groups.items():
                rows.append(dict(mag=mag, profile=profile, detector=det, case=case, k=k, devices=dv, detection=hit.loc[idx].mean()))
        for name, zc in ORS.items():
            each = TARGET_FPR / len(zc)
            cuts = [matched_cut(c[z], each) for z in zc]
            miss = lambda x: np.prod([1 - prob(x[z], cu, t) for z, (cu, t) in zip(zc, cuts)], axis=0)
            hit = pd.Series(1 - miss(a), index=a.index)
            for (case, k, dv), idx in a.groupby(['case', 'k', 'devices']).groups.items():
                rows.append(dict(mag=mag, profile=profile, detector=name, case=case, k=k, devices=dv, detection=hit.loc[idx].mean()))
    det = pd.DataFrame(rows); det.to_csv('pilot10_bp_detection.csv', index=False)
    order = single + list(ORS)
    KEYS = ['detector', 'case', 'mag', 'profile']
    ev = det[det.detection <= TARGET_FPR + MARGIN].sort_values(KEYS + ['k', 'detection'])
    best = ev.groupby(KEYS).first().reset_index()[KEYS + ['k', 'devices']].rename(columns={'k': 'cost', 'devices': 'cheapest_set'})
    cells = det[KEYS].drop_duplicates().merge(best, how='left'); cells.to_csv('pilot10_bp_cost.csv', index=False)
    ad = cells.groupby(['detector', 'case', 'mag']).cost.min().reset_index()
    pd.set_option('display.width', 200)
    print('\nTwin diagnostics (R2 of the twin on the calibration segment):'); print(pd.DataFrame(diags).round(3).to_string(index=False))
    print('\nBP detection at matched 10% FPR by number of compromised devices (mean over cases, SD multiples, profiles):')
    print(det.groupby(['k', 'detector']).detection.mean().unstack('detector')[order].round(2).to_string())
    print('\nBP detection with ONE compromised device (arterial line), by attack size in SD:')
    print(det[det.k == 1].groupby(['mag', 'detector']).detection.mean().unstack('detector')[order].round(2).to_string())
    print('\nBP attack cost, attacker free to choose the profile (evasion = detection <= 20%):')
    t = ad.groupby('detector').agg(evasion_found=('cost', lambda x: x.notna().mean()), mean_cost=('cost', 'mean')).round(2)
    print(t.loc[order].to_string())
    print('\nBP attack cost by attack size (mean cost / evasion found):')
    print(ad.groupby(['mag', 'detector']).cost.agg(['mean', lambda x: x.notna().mean()]).round(2).unstack('detector').to_string())


if __name__ == '__main__':
    main()
