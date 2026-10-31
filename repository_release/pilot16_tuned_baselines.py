"""Pilot 16: tuned learned baselines (Isolation Forest, autoencoder, LSTM forecaster).

Why. The baselines of Pilot 7 were small, untuned, normalised with one static mean and SD (so their score drifted
between calibration and test, giving 28-30% false alarms at a nominal 10%), and they were never given a fair chance.
This script gives each family a hyperparameter grid and a drift-robust option, selects the configuration WITHOUT
looking at any test window, and re-scores exactly the Pilot 12 and Pilot 14 attack windows.

Grid (per family): input normalisation 'static' (training mean and SD) or 'rolling' (each channel standardised
against its own median and MAD over [t-419 s, t-120 s], which is clean for a 60 s attack and removes slow drift);
Isolation Forest 100 or 300 trees; autoencoder hidden width 16 or 64 and 30 or 100 epochs; LSTM hidden size 32 or 64
and 15 or 40 epochs.

Selection (defender side, design cases only, calibration segment only). The calibration segment of each design
case is split in two halves. The first half sets the 90th-percentile threshold of the window score (target 10% false
alarms). In the second half the defender simulates unilateral tampering: 30 random windows per target with the
target source shifted by 1 patient SD (alternating signs, 10 s ramp). Configuration score = mean detection over the
four targets minus twice the excess of the false-alarm rate over 10% in the second half (a drift penalty). The best
configuration per family, averaged over the 8 design cases, is then used for all 20 cases.

Scoring. For tag in (pilot12, pilot14) the attack and control windows are read from {tag}_base_scores.csv (the
Pilot 7 output), rebuilt with the same offsets, scored with the chosen models, and written to
{tag}tuned_base_scores.csv with the same columns (red_target_score is kept), so pilot8_combined.py {tag}tuned works.
Thresholds for the det_* columns come from the whole calibration segment.

Usage:  .\\.venv\\Scripts\\python.exe pilot16_tuned_baselines.py      (about 30-40 minutes on CPU; run_tuned.ps1 does all steps)
Outputs: pilot16_validation.csv, pilot16_chosen.csv, pilot12tuned_base_scores.csv, pilot14tuned_base_scores.csv."""
import glob, itertools, json, os, time, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
from scipy.signal import lfilter
from sklearn.ensemble import IsolationForest
import pilot6 as p6
warnings.filterwarnings('ignore')
torch.set_num_threads(8)

ALLC = ['HR', 'PR', 'APR', 'SBP', 'MBP', 'DBP', 'EVM', 'NMBP', 'ET', 'ETP', 'RR', 'RRP', 'VRR', 'SPO2']
W = 20
PRE, L, A = p6.PRE, p6.L, p6.A
GAP, BASE_N, WARM = 120, 300, 420          # rolling baseline = median over the BASE_N s ending GAP s before t
TARGET_CH = {'HR': ['HR'], 'BP': ['SBP', 'MBP', 'DBP'], 'ETCO2': ['ET'], 'RR': ['RR']}
PRIMARY = {'HR': 'HR', 'BP': 'MBP', 'ETCO2': 'ET', 'RR': 'RR'}
N_VAL = 30
DESIGN_DIR = os.path.join('..', 'vitaldb_cases_ext')
REPL_DIR = os.path.join('cgptcode_expanded', 'vitaldb_cases_ext')
SETS = [('pilot12', DESIGN_DIR), ('pilot14', REPL_DIR)]
GRID = {
    'IF': [dict(n=n, norm=nm) for n in (100, 300) for nm in ('static', 'rolling')],
    'AE': [dict(h=h, ep=ep, norm=nm) for h in (16, 64) for ep in (30, 100) for nm in ('static', 'rolling')],
    'LSTM': [dict(h=h, ep=ep, norm=nm) for h in (32, 64) for ep in (15, 40) for nm in ('static', 'rolling')],
}
COLNAME = {'IF': 'IF_all', 'AE': 'AE', 'LSTM': 'LSTM'}


def ew(x):
    return lfilter([A], [1, -(1 - A)], x, axis=-1)


def feats(Z):
    return np.concatenate([Z, np.diff(Z, axis=-2, prepend=Z[..., :1, :])], axis=-1)


class AE(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        b = max(4, h // 4)
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, b), nn.ReLU(), nn.Linear(b, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        return self.net(x)


class Fore(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.rnn = nn.LSTM(d, h, batch_first=True); self.out = nn.Linear(h, d)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.out(o[:, -1])


def fit(model, X, Y, epochs, lr):
    torch.manual_seed(0)
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            opt.zero_grad(); loss = ((model(X[idx]) - Y[idx]) ** 2).mean(); loss.backward(); opt.step()
    return model.eval()


def prepare(path):
    D = p6.load(path); T = len(D['HR'])
    train_end, cal_end = int(T * 0.30), int(T * 0.70)
    X = pd.DataFrame({c: D[c] for c in ALLC}).ffill()
    X = X.fillna(X.iloc[:train_end].median()).fillna(0)
    Xv = X.values
    mu = Xv[:train_end].mean(0); sd = Xv[:train_end].std(0) + 1e-6
    med = X.rolling(BASE_N, min_periods=60).median()
    base = med.shift(GAP).values
    scale = (X - med).abs().rolling(BASE_N, min_periods=60).median().shift(GAP).values * 1.4826
    base = np.where(np.isnan(base), mu, base)
    scale = np.maximum(np.where(np.isnan(scale), sd, scale), 0.2 * sd)
    Z = {'static': np.clip((Xv - mu) / sd, -10, 10), 'rolling': np.clip((Xv - base) / scale, -10, 10)}
    return dict(D=D, T=T, train_end=train_end, cal_end=cal_end, Xv=Xv, mu=mu, sd=sd, base=base, scale=scale, Z=Z)


def to_z(P, norm, Xseg, sl):
    if norm == 'static':
        return np.clip((Xseg - P['mu']) / P['sd'], -10, 10)
    return np.clip((Xseg - P['base'][sl]) / P['scale'][sl], -10, 10)


def fit_family(P, fam, cfg):
    Z = P['Z'][cfg['norm']]
    lo = WARM if cfg['norm'] == 'rolling' else 0
    Ztr = Z[lo:P['train_end']]
    if fam == 'IF':
        return fam, IsolationForest(n_estimators=cfg['n'], random_state=0).fit(feats(Ztr)), None
    if fam == 'AE':
        F = feats(Ztr)
        m = fit(AE(F.shape[1], cfg['h']), F, F, cfg['ep'], 1e-3)
        with torch.no_grad():
            sc = np.sqrt(((m(torch.tensor(F, dtype=torch.float32)).numpy() - F) ** 2).mean(0)) + 1e-3
        return fam, m, sc
    idx = np.arange(W, len(Ztr))
    Xw = np.stack([Ztr[i - W:i] for i in idx]); Y = Ztr[idx]
    m = fit(Fore(Z.shape[1], cfg['h']), Xw, Y, cfg['ep'], 2e-3)
    with torch.no_grad():
        sc = np.sqrt(((m(torch.tensor(Xw[::2], dtype=torch.float32)).numpy() - Y[::2]) ** 2).mean(0)) + 1e-3
    return fam, m, sc


def window_scores(model, Zs):
    """Zs: (n, W+PRE+L, c) -> maximum EWMA of the per-step score over the last L steps, shape (n,)."""
    fam, m, sc = model
    n, st, c = Zs.shape
    if fam == 'IF':
        Fs = feats(Zs)
        step = (-m.score_samples(Fs.reshape(-1, Fs.shape[-1]))).reshape(n, st)
    elif fam == 'AE':
        Fs = feats(Zs)
        with torch.no_grad():
            rec = m(torch.tensor(Fs.reshape(-1, Fs.shape[-1]), dtype=torch.float32)).numpy().reshape(Fs.shape)
        step = (((rec - Fs) / sc) ** 2).mean(-1)
    else:
        step = np.zeros((n, st)); t = torch.tensor(Zs, dtype=torch.float32); s_t = torch.tensor(sc, dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, n, 128):
                blk = t[i:i + 128]
                win = blk.unfold(1, W, 1)[:, :-1].permute(0, 1, 3, 2)
                b, mm = win.shape[:2]
                pred = m(win.reshape(b * mm, W, c)).reshape(b, mm, c)
                step[i:i + 128, W:] = (((pred - blk[:, W:]) / s_t) ** 2).mean(-1).numpy()
    return ew(step[:, W:])[:, -L:].max(1)


def clean_maxima(P, model, norm, a0, a1):
    starts = list(range(a0, a1 - L + 1, L))
    Z = P['Z'][norm]
    segs = np.stack([Z[s - PRE - W:s + L] for s in starts])
    return window_scores(model, segs)


def validate(P, case, model, cfg):
    train_end, cal_end = P['train_end'], P['cal_end']
    mid = (train_end + cal_end) // 2
    norm = cfg['norm']
    thr = np.quantile(clean_maxima(P, model, norm, train_end, mid), 0.90)
    fpr_b = float(np.mean(clean_maxima(P, model, norm, mid, cal_end) > thr))
    rng = np.random.default_rng(1000 + case)
    ramp = p6.PROFILES['ramp_10s']
    tprs = {}
    for t, chans in TARGET_CH.items():
        prim = PRIMARY[t]
        valid = np.isfinite(P['D'][prim])
        starts = [s for s in range(mid + PRE + W, cal_end - L - 1) if valid[s:s + L].all()]
        if len(starts) < N_VAL:
            continue
        sd_t = float(np.nanstd(P['D'][prim][:train_end]))
        pick = rng.choice(starts, N_VAL, replace=False)
        segs = []
        for j, s in enumerate(pick):
            sl = slice(s - PRE - W, s + L)
            seg = P['Xv'][sl].copy()
            for ch in chans:
                seg[W + PRE:, ALLC.index(ch)] += (1 if j % 2 == 0 else -1) * sd_t * ramp
            segs.append(to_z(P, norm, seg, sl))
        tprs[t] = float(np.mean(window_scores(model, np.stack(segs)) > thr))
    mean_tpr = float(np.mean(list(tprs.values()))) if tprs else np.nan
    return dict(tpr_mean=mean_tpr, fpr_b=fpr_b, score=mean_tpr - 2 * max(0.0, fpr_b - 0.10), **{f'tpr_{k}': v for k, v in tprs.items()})


def case_files(directory):
    return [f for f in sorted(glob.glob(os.path.join(directory, 'case*.csv'))) if not f.endswith('_wave.csv')]


def phase1():
    rows = []
    for f in case_files(DESIGN_DIR):
        case = int(os.path.basename(f)[4:-4]); t0 = time.time()
        P = prepare(f)
        for fam, cfgs in GRID.items():
            for cfg in cfgs:
                m = fit_family(P, fam, cfg)
                rows.append(dict(case=case, family=fam, cfg=json.dumps(cfg, sort_keys=True), **validate(P, case, m, cfg)))
        print(f'validation, case {case}: {time.time() - t0:.0f}s', flush=True)
    val = pd.DataFrame(rows); val.to_csv('pilot16_validation.csv', index=False)
    mean = val.groupby(['family', 'cfg'])[['tpr_mean', 'fpr_b', 'score']].mean().reset_index()
    chosen = mean.sort_values('score', ascending=False).groupby('family').head(1)
    chosen.to_csv('pilot16_chosen.csv', index=False)
    pd.set_option('display.width', 200, 'display.max_colwidth', 80)
    print('\nValidation (design cases, calibration segment only), mean over cases; the best row per family is chosen:')
    print(mean.sort_values(['family', 'score'], ascending=[True, False]).round(3).to_string(index=False))
    return {r.family: json.loads(r.cfg) for r in chosen.itertuples()}


def phase2(chosen):
    devmap = {t: {d[0]: d[1] for d in tg['devs']} for t, tg in p6.TARGETS.items()}
    for tag, directory in SETS:
        src = f'{tag}_base_scores.csv'
        if not os.path.exists(src):
            print('skipped (missing):', src); continue
        old = pd.read_csv(src); parts = []
        for f in case_files(directory):
            case = int(os.path.basename(f)[4:-4]); sub = old[old.case == case].copy()
            if sub.empty:
                continue
            t0 = time.time(); P = prepare(f)
            cal_starts = (P['train_end'], P['cal_end'])
            for fam, cfg in chosen.items():
                m = fit_family(P, fam, cfg); norm = cfg['norm']
                thr = np.quantile(clean_maxima(P, m, norm, *cal_starts), 0.90)
                Zs = np.empty((len(sub), W + PRE + L, len(ALLC)))
                for j, (_, r) in enumerate(sub.iterrows()):
                    s = int(r.start_index); sl = slice(s - PRE - W, s + L)
                    seg = P['Xv'][sl].copy()
                    if r.k > 0:
                        delta = r.offset * p6.PROFILES[r.profile]
                        for dev in r.devices.split('+'):
                            for ch in devmap[r.target][dev]:
                                seg[W + PRE:, ALLC.index(ch)] += delta
                    Zs[j] = to_z(P, norm, seg, sl)
                sc = window_scores(m, Zs)
                sub[f'score_{COLNAME[fam]}'] = sc
                sub[f'det_{COLNAME[fam]}'] = sc > thr
            parts.append(sub); print(f'{tag}, case {case}: {time.time() - t0:.0f}s', flush=True)
        new = pd.concat(parts); out = f'{tag}tuned_base_scores.csv'; new.to_csv(out, index=False)
        ctrl_old, ctrl_new = old[old.k == 0], new[new.k == 0]
        tab = pd.DataFrame({
            f'{n} old': ctrl_old.groupby('target')[f'det_{c}'].mean() for n, c in (('IF', 'IF_all'), ('AE', 'AE'), ('LSTM', 'LSTM'))
        }).join(pd.DataFrame({f'{n} tuned': ctrl_new.groupby('target')[f'det_{c}'].mean() for n, c in (('IF', 'IF_all'), ('AE', 'AE'), ('LSTM', 'LSTM'))}))
        print(f'\n{tag}: realised false-alarm rate on clean test windows (nominal 10%), old versus tuned detectors:')
        print(tab.round(2).to_string()); print('wrote', out)


if __name__ == '__main__':
    phase2(phase1())
