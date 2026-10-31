"""Pilot 7: learned baselines on exactly the attack windows of Pilot 6.

Detectors compared with the redundancy detector (scores taken from pilot6_results.csv):
  IF_all  Isolation Forest on all device channels (values and first differences), not only the 8 of Pilot 5/6
  AE      MLP autoencoder on the same features
  LSTM    LSTM forecaster: predicts the next standardised vector of all channels from the last 20 steps
All three see every channel the redundancy detector sees (HR, pleth PR, arterial-waveform PR, SBP/MBP/DBP,
EV1000 MBP, cuff MBP, EtCO2 from both devices, RR from both devices, ventilator RR, SpO2), so the comparison
asks whether a learned model can do without hand-written cross-device invariants.

Protocol is the same as Pilot 5/6: first 30% of each case trains, the next 40% calibrates the threshold
(90th percentile of the maximum EWMA score over clean non-overlapping 60 s windows, i.e. 10% target false
alarms), the last 30% is the test segment. The attack windows (case, target, mag, profile, k, devices,
start_index) are read from pilot6_results.csv and rebuilt with the same offsets, so every detector scores the
same windows. Score of a window = maximum EWMA (0.1) of the per-step anomaly score over the 60 attacked steps.

Usage:  .\\.venv\\Scripts\\python.exe pilot7_baselines.py        (needs torch; about 10-20 minutes on CPU)
Outputs: pilot7_scores.csv (one row per window), pilot7_summary.csv, pilot7_matched_fpr.csv, pilot7_auc.csv."""
import glob, os, sys, time, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
from scipy.signal import lfilter
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import pilot6 as p6
warnings.filterwarnings('ignore')
torch.set_num_threads(8)

TAG = sys.argv[1] if len(sys.argv) > 1 else None          # e.g. pilot9: read pilot9_results.csv, write pilot9_base_*.csv
IN = f'{TAG}_results.csv' if TAG else 'pilot6_results.csv'
OUTP = f'{TAG}_base' if TAG else 'pilot7'
ALLC = ['HR', 'PR', 'APR', 'SBP', 'MBP', 'DBP', 'EVM', 'NMBP', 'ET', 'ETP', 'RR', 'RRP', 'VRR', 'SPO2']
W = 20                      # LSTM history
PRE, L, A = p6.PRE, p6.L, p6.A
KEYS = ['case', 'target', 'mag', 'profile', 'k', 'devices', 'start_index']
DETS = ['IF_all', 'AE', 'LSTM']
EPOCHS_AE, EPOCHS_LSTM = 30, 15


def ew(x, axis=-1):
    return lfilter([A], [1, -(1 - A)], x, axis=axis)


class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(),
                                 nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, x):
        return self.net(x)


class Fore(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.rnn = nn.LSTM(d, 48, batch_first=True)
        self.out = nn.Linear(48, d)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h[:, -1])


def feats(Z):
    """Z: (..., n, c) -> values and first differences (first difference zero)."""
    d = np.diff(Z, axis=-2, prepend=Z[..., :1, :])
    return np.concatenate([Z, d], axis=-1)


def fit(model, X, Y, epochs, lr):
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            opt.zero_grad(); loss = ((model(X[idx]) - Y[idx]) ** 2).mean(); loss.backward(); opt.step()
    return model.eval()


def run_case(path, res):
    case = int(os.path.basename(path)[4:-4])
    sub = res[res.case == case].copy()
    if sub.empty:
        return None
    torch.manual_seed(0)
    D = p6.load(path); T = len(D['HR'])
    train_end, cal_end = int(T * 0.30), int(T * 0.70)
    X = pd.DataFrame({c: D[c] for c in ALLC}).ffill()
    X = X.fillna(X.iloc[:train_end].median()).fillna(0)
    Xv = X.values; mu = Xv[:train_end].mean(0); sd = Xv[:train_end].std(0) + 1e-6
    Z = np.clip((Xv - mu) / sd, -10, 10)
    Ztr = Z[:train_end]
    # models trained on the first 30%
    Ftr = feats(Ztr)
    ae = fit(AE(Ftr.shape[1]), Ftr, Ftr, EPOCHS_AE, 1e-3)
    idx = np.arange(W, len(Ztr))
    lstm = fit(Fore(len(ALLC)), np.stack([Ztr[i - W:i] for i in idx]), Ztr[idx], EPOCHS_LSTM, 2e-3)
    iso = IsolationForest(n_estimators=100, random_state=0).fit(Ftr)

    def step_scores(Zs):
        """Zs: (n, steps, c) -> dict of (n, steps) per-step anomaly scores (first W steps are warm-up)."""
        n, st, c = Zs.shape
        Fs = feats(Zs)
        out = {}
        with torch.no_grad():
            rec = ae(torch.tensor(Fs.reshape(-1, Fs.shape[-1]), dtype=torch.float32)).numpy().reshape(Fs.shape)
            out['AE'] = (((rec - Fs) / ae_scale) ** 2).mean(-1)
            ls = np.zeros((n, st))
            t = torch.tensor(Zs, dtype=torch.float32)
            for i in range(0, n, 128):
                blk = t[i:i + 128]
                win = blk.unfold(1, W, 1)[:, :-1].permute(0, 1, 3, 2)          # (b, st-W, W, c) ends before target
                b, m = win.shape[:2]
                pred = lstm(win.reshape(b * m, W, c)).reshape(b, m, c)
                ls[i:i + 128, W:] = (((pred - blk[:, W:]) / lstm_scale) ** 2).mean(-1).numpy()
        out['LSTM'] = ls
        out['IF_all'] = (-iso.score_samples(Fs.reshape(-1, Fs.shape[-1]))).reshape(n, st)
        return out

    # per-feature error scale from the training segment so that no channel dominates the mean
    with torch.no_grad():
        Ftr_t = torch.tensor(Ftr, dtype=torch.float32)
        ae_scale = np.sqrt(((ae(Ftr_t).numpy() - Ftr) ** 2).mean(0)) + 1e-3
        pw = np.stack([Ztr[i - W:i] for i in idx[::2]])
        lstm_scale = np.sqrt(((lstm(torch.tensor(pw, dtype=torch.float32)).numpy() - Ztr[idx[::2]]) ** 2).mean(0)) + 1e-3

    # calibration: maximum EWMA over clean non-overlapping 60 s windows in the calibration segment
    cal_starts = list(range(train_end, cal_end - L + 1, L))
    segs = np.stack([Z[s - PRE - W:s + L] for s in cal_starts if s - PRE - W >= 0])
    cs = step_scores(segs)
    thr = {d: np.quantile(ew(cs[d][:, W:])[:, -L:].max(1), 0.90) for d in DETS}

    # attack windows exactly as in Pilot 6
    devmap = {t: {d[0]: d[1] for d in tg['devs']} for t, tg in p6.TARGETS.items()}
    n = len(sub)
    Zs = np.empty((n, W + PRE + L, len(ALLC)))
    for j, (_, r) in enumerate(sub.iterrows()):
        s = int(r.start_index)
        seg = Xv[s - PRE - W:s + L].copy()
        if r.k > 0:
            delta = (r.offset if 'offset' in sub.columns else r.mag) * p6.PROFILES[r.profile]
            for dev in r.devices.split('+'):
                for ch in devmap[r.target][dev]:
                    seg[W + PRE:, ALLC.index(ch)] += delta
        Zs[j] = np.clip((seg - mu) / sd, -10, 10)
    sc = step_scores(Zs)
    for d in DETS:
        e = ew(sc[d][:, W:])[:, -L:]
        sub[f'score_{d}'] = e.max(1)
        sub[f'det_{d}'] = sub[f'score_{d}'] > thr[d]
    return sub


def main():
    res = pd.read_csv(IN)
    keep = KEYS + (['offset'] if 'offset' in res.columns else []) + ['red_target_score', 'iso_target_score', 'det_red_target', 'det_iso_target', 'kmax']
    res = res[keep]
    parts = []
    for f in sorted(glob.glob(os.path.join(p6.DATA, 'case*.csv'))):
        if f.endswith('_wave.csv'):
            continue
        t0 = time.time(); out = run_case(f, res)
        if out is not None:
            parts.append(out)
        print(f'{os.path.basename(f)}: done in {time.time() - t0:.0f}s', flush=True)
    d = pd.concat(parts); d.to_csv(f'{OUTP}_scores.csv', index=False)

    detectors = {'Redundancy': ('red_target_score', 'det_red_target'), 'IF (8 ch)': ('iso_target_score', 'det_iso_target'),
                 'IF (all ch)': ('score_IF_all', 'det_IF_all'), 'Autoencoder': ('score_AE', 'det_AE'),
                 'LSTM forecaster': ('score_LSTM', 'det_LSTM')}
    rows = []
    for (case, target), g in d.groupby(['case', 'target']):
        c, a, s1 = g[g.k == 0], g[g.k > 0], g[g.k == 1]
        for name, (sc, dt) in detectors.items():
            auc = lambda att: roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(att))], np.r_[c[sc], att[sc]])
            rows.append(dict(case=case, target=target, detector=name, fpr=c[dt].mean(), tpr_single=s1[dt].mean(),
                             auc_all=auc(a), auc_single=auc(s1)))
    case_tab = pd.DataFrame(rows)
    summ = case_tab.groupby(['target', 'detector'])[['fpr', 'tpr_single', 'auc_all', 'auc_single']].mean().reset_index()
    summ.to_csv(f'{OUTP}_summary.csv', index=False)
    case_tab.to_csv(f'{OUTP}_case_metrics.csv', index=False)
    # matched-FPR detection at 10% per (target, magnitude, profile), controls pooled over cases as in Pilot 5
    mrows = []
    for (target, mag, profile), c in d[d.k == 0].groupby(['target', 'mag', 'profile']):
        a = d[(d.target == target) & (d.mag == mag) & (d.profile == profile) & (d.k > 0)]
        for name, (sc, _) in detectors.items():
            cut, tp, fpr = p6.matched_cut(c[sc], 0.10)
            for k, ak in a.groupby('k'):
                mrows.append(dict(target=target, mag=mag, profile=profile, detector=name, k=k, fpr=fpr,
                                  detection=p6.matched_detection(ak[sc], cut, tp)))
    m = pd.DataFrame(mrows); m.to_csv(f'{OUTP}_matched_fpr.csv', index=False)
    pd.set_option('display.width', 180)
    print('\nCase-mean metrics (thresholds from the calibration segment, 10% target FPR):')
    print(summ.round(3).to_string(index=False))
    print('\nDetection at matched 10% test FPR, by number of compromised devices (mean over magnitudes and profiles):')
    print(m.groupby(['target', 'k', 'detector']).detection.mean().unstack('detector').round(3).to_string())


if __name__ == '__main__':
    main()
