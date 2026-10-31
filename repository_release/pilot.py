"""Pilot: does physiological analytical redundancy (ECG heart rate vs pleth pulse rate) expose
FDI attacks that learned/statistical detectors miss, and what does an attacker need to evade it?"""
import numpy as np, pandas as pd, glob
from sklearn.ensemble import IsolationForest
rng = np.random.default_rng(0)
VALID = {'Temp': (30, 43), 'SpO2': (50, 100), 'Pulse_Rate': (25, 220), 'SYS': (40, 250), 'DIA': (15, 150),
         'Heart_rate': (25, 220), 'Resp_Rate': (4, 60), 'ST': (-5, 5)}
CH = list(VALID)
def load(f):
    d = pd.read_csv(f)[CH].astype(float)
    for c, (lo, hi) in VALID.items(): d.loc[(d[c] < lo) | (d[c] > hi), c] = np.nan
    return d.interpolate(limit_direction='both').values

def ewma(x, a=0.1):
    y = np.zeros_like(x); m = 0.0
    for i, v in enumerate(x): m = (1 - a) * m + a * v; y[i] = m
    return y

def attack(X, s, L, kind, mag, coord):
    """inject on Heart_rate (col 5); coord=True also moves Pulse_Rate (col 2) identically"""
    Y = X.copy(); t = np.arange(L)
    ramp = np.minimum(t / (L / 3), 1.0)                       # reach full magnitude after L/3 samples
    delta = mag * ramp if kind == 'drift' else np.full(L, mag)
    Y[s:s + L, 5] += delta
    if coord: Y[s:s + L, 2] += delta
    return Y

rows = []
for f in sorted(glob.glob('../vitaldb_cases/*.csv')):
    case = f.split('case')[-1][:-4]; X = load(f); T = len(X); n = T // 2
    mu, sd = X[:n].mean(0), X[:n].std(0) + 1e-6
    # redundancy residual, robust-standardised on the training half
    r = X[:, 5] - X[:, 2]; med = np.median(r[:n]); mad = 1.4826 * np.median(np.abs(r[:n] - med)) + 0.5
    def red_score(Y):
        rr = (Y[:, 5] - Y[:, 2] - med) / mad; return np.abs(ewma(np.clip(rr, -20, 20)))
    # statistical/learned baseline: Isolation Forest on standardised values + first differences
    def feats(Y): Z = (Y - mu) / sd; return np.hstack([Z, np.vstack([np.zeros(8), np.diff(Z, axis=0)])])
    iso = IsolationForest(n_estimators=200, random_state=0).fit(feats(X[:n]))
    def iso_score(Y): return ewma(-iso.score_samples(feats(Y)))
    # thresholds at 1% false-alarm rate on clean test half
    thr_red = np.quantile(red_score(X)[n:], 0.99); thr_iso = np.quantile(iso_score(X)[n:], 0.99)
    for kind in ['drift', 'bias']:
        for mag in [3, 5, 10, 20]:
            for coord in [False, True]:
                det_r = det_i = 0; K = 15
                for k in range(K):
                    L = 60; s = int(rng.integers(n + 50, T - L - 50))
                    Y = attack(X, s, L, kind, mag, coord)
                    det_r += (red_score(Y)[s:s + L] > thr_red).any()
                    det_i += (iso_score(Y)[s:s + L] > thr_iso).any()
                rows.append(dict(case=case, kind=kind, mag=mag, coord=coord, red=det_r / K, iso=det_i / K, mad=mad))
d = pd.DataFrame(rows); d.to_csv('pilot_results.csv', index=False)
print(d.groupby(['coord', 'kind', 'mag'])[['red', 'iso']].mean().round(2).to_string())
print('\nper-patient MAD of HR-PR residual (bpm):', d.groupby('case').mad.first().round(2).to_dict())
