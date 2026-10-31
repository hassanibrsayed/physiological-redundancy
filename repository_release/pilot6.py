"""Pilot 6 = Pilot 5 plus R2: pulse rate derived from the arterial waveform (APR, from download_wave.py) as a
third heart-rate device. R2 = HR(ECG) - APR, and the HR target gains a third device (ART_WAVE). Everything else
(split, profiles, thresholds, Isolation Forest baseline) is unchanged so the results compare with Pilot 5.

Pilot 5: independent calibration and learned rate-of-change constraints.

This version keeps earlier pilot scripts and outputs intact. It uses the first 30% of
each case for fitting, the next 40% for threshold calibration, and the final 30% for
attack evaluation. Thresholds are estimated from clean calibration windows only;
the realized false-positive rate is measured on the final segment without retuning.
Results remain exploratory because splits are within-case and only eight cases are
available.

Multi-invariant redundancy detector vs Isolation Forest, with detection reported against the
number of compromised devices (attack cost).

Targets: HR, BP (arterial MBP), EtCO2, RR. For each target the attacker falsifies the target channel and then
adds partner devices one at a time (k = 1, 2, ...). Each device's channels receive a shared step, 10-second
ramp, or 60-second drift, preserving within-device relationships while testing onset sensitivity.

Invariants used (R2, the pulse rate from the arterial waveform, is not included):
  R1  HR(ECG) - pulse rate (pleth)                 R2b ART_MBP - EV1000 ART_MBP (cases that have EV1000)
  R3  ART_MBP - NIBP_MBP (only while the cuff value is present)
  R4  EtCO2 monitor - EtCO2 anaesthesia machine    R5  RR(capnography) - ventilator RR (controlled ventilation only)
  R5b RR monitor - RR anaesthesia machine
  C1  SBP > MBP > DBP   C2  MBP = DBP + (SBP-DBP)/3 (per-patient offset)   C3  15 <= pulse pressure <= 100
  C4  Per-channel rate-of-change score scaled by the training 99.5th percentile
Thresholds: target-specific redundancy and Isolation Forest thresholds target a 10% clean 60-second-window
false-alarm rate using calibration data only. The final 30% is test data and is not used to retune thresholds.
Attacks outside channel ranges are omitted. Writes pilot6_results.csv plus calibration, ROC, and cost summaries."""
import glob, itertools, os, time, warnings
import numpy as np, pandas as pd
from scipy.signal import lfilter
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

DATA = os.environ.get('DATA_DIR', os.path.join('..', 'vitaldb_cases_ext'))   # DATA_DIR selects another case folder
rng = np.random.default_rng(0)
L, K, A, PRE = 60, 15, 0.1, 100        # attack length, attacks per config, EWMA weight, warm-up samples
TARGET_FPR = 0.10                       # shared operating point; 1% is not estimable with this calibration size
PROFILES = {
    'step': np.ones(L),
    'ramp_10s': np.minimum(np.arange(1, L + 1) / 10, 1.0),
    'drift_60s': np.arange(1, L + 1) / L,
}
RANGE_REJECTIONS = []
KT = {'BP': 40}                         # more BP attacks: only about a quarter of windows overlap a cuff reading

NAMES = {'HR': 'Solar8000/HR', 'PR': 'Solar8000/PLETH_HR', 'SPO2': 'Solar8000/PLETH_SPO2',
         'SBP': 'Solar8000/ART_SBP', 'MBP': 'Solar8000/ART_MBP', 'DBP': 'Solar8000/ART_DBP',
         'NMBP': 'Solar8000/NIBP_MBP', 'EVM': 'EV1000/ART_MBP',
         'ET': 'Solar8000/ETCO2', 'ETP': 'Primus/ETCO2',
         'RR': 'Solar8000/RR_CO2', 'RRP': 'Primus/RR_CO2', 'VRR': 'Solar8000/VENT_RR',
         'APR': None}   # APR is read from case<ID>_artpr.csv in load()
RANGE = {'HR': (25, 220), 'PR': (25, 220), 'APR': (25, 220), 'SPO2': (50, 100), 'SBP': (40, 250), 'MBP': (20, 200), 'DBP': (15, 150),
         'NMBP': (20, 200), 'EVM': (20, 200), 'ET': (5, 80), 'ETP': (5, 80), 'RR': (2, 60), 'RRP': (2, 60), 'VRR': (0, 60)}
FILL = {'NMBP': 0, 'ETP': 10, 'RRP': 10}   # forward-fill limit; default 3 samples, cuff values are not held
IFC = ['HR', 'PR', 'SBP', 'MBP', 'DBP', 'ET', 'RR', 'SPO2']

# name: (residual, needed channels, kind, MAD floor)
INV = {
    'R1': (lambda D: D['HR'] - D['PR'], ['HR', 'PR'], 'ewma', 0.5),
    'R2': (lambda D: D['HR'] - D['APR'], ['HR', 'APR'], 'ewma', 0.5),
    'R2b': (lambda D: D['MBP'] - D['EVM'], ['MBP', 'EVM'], 'ewma', 1.0),
    # cuff reading vs the arterial MBP averaged since the reading appeared; per-patient offset and spread learned
    'R3': (lambda D: run_cummean(D['MBP'] - D['NMBP']), ['MBP', 'NMBP'], 'raw', 2.0),
    'R4': (lambda D: D['ET'] - D['ETP'], ['ET', 'ETP'], 'ewma', 1.0),
    'R5': (lambda D: np.where((D['VRR'] > 0) & (D['RR'] > 0), D['RR'] - D['VRR'], np.nan), ['RR', 'VRR'], 'ewma', 1.0),
    'R5b': (lambda D: D['RR'] - D['RRP'], ['RR', 'RRP'], 'ewma', 1.0),
    'C1': (lambda D: np.maximum(np.maximum(D['DBP'] - D['MBP'], D['MBP'] - D['SBP']), 0), ['SBP', 'MBP', 'DBP'], 'hinge', 0),
    'C2': (lambda D: D['MBP'] - (D['DBP'] + (D['SBP'] - D['DBP']) / 3), ['SBP', 'MBP', 'DBP'], 'ewma', 1.0),
    'C3': (lambda D: np.maximum(np.maximum(15 - (D['SBP'] - D['DBP']), (D['SBP'] - D['DBP']) - 100), 0), ['SBP', 'DBP'], 'hinge', 0),
}
RATE_FLOORS = {'HR': 0.2, 'PR': 0.2, 'APR': 0.2, 'SBP': 0.5, 'MBP': 0.5, 'DBP': 0.5,
               'ET': 0.2, 'ETP': 0.2, 'RR': 0.2, 'RRP': 0.2, 'VRR': 0.2}
for channel, floor in RATE_FLOORS.items():
    INV[f'C4_{channel}'] = (
        lambda D, c=channel: np.r_[np.nan, np.diff(D[c])],
        [channel], 'rate', floor)
TARGETS = {
    'HR': dict(primary='HR', mags=[5, 10, 20], devs=[('ECG', ['HR']), ('PLETH', ['PR']), ('ART_WAVE', ['APR'])]),
    'BP': dict(primary='MBP', mags=[10, 20, 30, 40], devs=[('ART', ['SBP', 'MBP', 'DBP']), ('NIBP', ['NMBP']), ('EV1000', ['EVM'])]),
    'ETCO2': dict(primary='ET', mags=[5, 10], devs=[('MON_CAPNO', ['ET']), ('MACHINE_CAPNO', ['ETP'])]),
    'RR': dict(primary='RR', mags=[4, 8], devs=[('MON_CAPNO', ['RR']), ('VENT', ['VRR']), ('MACHINE_CAPNO', ['RRP'])]),
}
TARGET_INVS = {
    'HR': {'R1', 'R2', 'C4_HR', 'C4_PR', 'C4_APR'},
    'BP': {'R2b', 'R3', 'C1', 'C2', 'C3', 'C4_SBP', 'C4_MBP', 'C4_DBP'},
    'ETCO2': {'R4', 'C4_ET', 'C4_ETP'},
    'RR': {'R5', 'R5b', 'C4_RR', 'C4_RRP', 'C4_VRR'},
}


def ew(x):
    return lfilter([A], [1, -(1 - A)], x)


def run_cummean(x):
    """Causal running mean inside each contiguous run of valid samples (one cuff reading = one run)."""
    ok = np.isfinite(x); v = np.where(ok, x, 0.0)
    c = np.cumsum(v); k = np.cumsum(ok)
    start = ok & ~np.r_[False, ok[:-1]]
    idx = np.maximum.accumulate(np.where(start, np.arange(len(x)), 0))
    num = c - (c - v)[idx]; den = k - (k - ok)[idx]
    return np.where(ok, num / np.maximum(den, 1), np.nan)


def load(path):
    raw = pd.read_csv(path)
    D = {}
    apr_path = os.path.join(os.path.dirname(path), 'artpr', os.path.basename(path))
    apr = pd.read_csv(apr_path)['ART_PR'].reindex(range(len(raw))).values if os.path.exists(apr_path) else None
    for k, col in NAMES.items():
        if k == 'APR':
            s = pd.Series(apr if apr is not None else np.nan, index=raw.index, dtype=float)
        else:
            s = raw[col].astype(float) if col in raw else pd.Series(np.nan, index=raw.index)
        lo, hi = RANGE[k]
        s = s.where((s >= lo) & (s <= hi))
        lim = FILL.get(k, 3)
        D[k] = (s.ffill(limit=lim) if lim else s).values
    return D


def inv_scores(D, par):
    out = {}
    for name, (fn, _, kind, _) in INV.items():
        if name not in par:
            continue
        res = fn(D)
        if kind == 'hinge':
            s = np.nan_to_num(res)
        else:
            med, mad = par[name]
            r = np.nan_to_num(np.clip((res - med) / mad, -20, 20))
            s = np.abs(ew(r)) if kind == 'ewma' else np.abs(r)
        out[name] = s
    return out


def within_channel_ranges(D, channels, delta):
    """Keep injected values inside the documented broad physiological ranges."""
    for channel in channels:
        values = D[channel][PRE:PRE + L] + delta
        finite = np.isfinite(values)
        lo, hi = RANGE[channel]
        if finite.any() and ((values[finite] < lo).any() or (values[finite] > hi).any()):
            return False
    return True


def matched_cut(scores, target_fpr):
    """Choose a clean-control order-statistic cut and randomize ties to hit FPR."""
    values = np.asarray(scores, dtype=float)
    rank = max(0, min(len(values) - 1, int(np.ceil((1 - target_fpr) * len(values))) - 1))
    cut = np.sort(values)[rank]
    above = np.mean(values > cut)
    tied = np.mean(values == cut)
    tie_probability = np.clip((target_fpr - above) / tied, 0.0, 1.0) if tied else 0.0
    return cut, tie_probability, above + tie_probability * tied


def matched_detection(scores, cut, tie_probability):
    """Expected detection under the same randomized tie rule as clean controls."""
    values = np.asarray(scores, dtype=float)
    return np.mean(values > cut) + tie_probability * np.mean(values == cut)


def run_case(path):
    case = os.path.basename(path)[4:-4]
    D = load(path); T = len(D['HR'])
    train_end = int(T * 0.30)
    cal_end = int(T * 0.70)
    # active invariants: every needed channel valid in at least 5% of the training half
    ok = {k: np.isfinite(D[k][:train_end]).mean() > 0.05 for k in D}
    par = {}
    for name, (fn, need, kind, floor) in INV.items():
        if not all(ok[c] for c in need):
            continue
        res = fn(D)[:train_end]; res = res[np.isfinite(res)]
        if kind == 'hinge':
            par[name] = None
        elif len(res) < 30:
            continue
        elif kind == 'rate':
            # Learn a robust per-channel rate scale from training data only.
            scale = max(float(np.quantile(np.abs(res), 0.995)), floor)
            par[name] = (0.0, scale)
        else:
            med = np.median(res); mad = max(1.4826 * np.median(np.abs(res - med)), INV[name][3])
            par[name] = (med, mad)
    if not par:
        return []
    # Use non-overlapping clean calibration windows. Each per-invariant threshold
    # targets 1% / number-of-invariants at the window level (union-bound control).
    cal_starts = list(range(train_end, cal_end - L + 1, L))
    if len(cal_starts) < 10:
        return []
    calibration_maxima = {}
    for name in par:
        residual = INV[name][0](D)
        values = [np.nanmax(inv_scores(D, {name: par[name]})[name][st:st + L])
                  for st in cal_starts
                  if np.isfinite(residual[st:st + L]).any()]
        if len(values) >= 10:
            calibration_maxima[name] = values
    # An invariant without enough calibration windows is excluded explicitly;
    # this matters for intermittent cuff readings (R3).
    par = {name: par[name] for name in calibration_maxima}
    if not par:
        return []
    clean = inv_scores(D, par)
    q = 1 - 0.01 / len(par)
    thr = {}
    for name, maxima in calibration_maxima.items():
        thr[name] = np.quantile(maxima, q)
    # Isolation Forest baseline (same recipe as the first pilot)
    X = pd.DataFrame({c: D[c] for c in IFC}).ffill()
    # Fill remaining leading gaps with training-only medians; never backfill from future data.
    X = X.fillna(X.iloc[:train_end].median()).fillna(0)
    Xf = {c: X[c].values.copy() for c in IFC}
    mu = X.iloc[:train_end].mean().values; sd = X.iloc[:train_end].std().values + 1e-6

    def feats(M):
        Z = (M - mu) / sd
        return np.hstack([Z, np.vstack([np.zeros(Z.shape[1]), np.diff(Z, axis=0)])])
    iso = IsolationForest(n_estimators=100, random_state=0).fit(feats(X.values[:train_end]))
    iso_scores = ew(-iso.score_samples(feats(X.values)))
    iso_cal_max = [np.max(iso_scores[st:st + L]) for st in cal_starts]
    thr_iso = np.quantile(iso_cal_max, 0.99) if len(iso_cal_max) >= 10 else np.inf

    rows = []
    for tname, tg in TARGETS.items():
        prim = tg['primary']
        target_inv = [name for name in par if name in TARGET_INVS[tname]]
        if not target_inv:
            continue
        # Calibrate the target-specific detector at a 10% window-level FPR.
        # Bonferroni correction controls the OR across that target's invariants.
        target_q = 1 - TARGET_FPR / len(target_inv)
        target_thr = {name: np.quantile(calibration_maxima[name], target_q)
                      for name in target_inv}
        target_inv_no_c4 = [name for name in target_inv if not name.startswith('C4_')]
        target_q_no_c4 = 1 - TARGET_FPR / len(target_inv_no_c4)
        target_thr_no_c4 = {name: np.quantile(calibration_maxima[name], target_q_no_c4)
                            for name in target_inv_no_c4}
        iso_target_thr = np.quantile(iso_cal_max, 1 - TARGET_FPR)
        devs = [d for i, d in enumerate(tg['devs']) if i == 0 or all(ok[c] for c in d[1])]
        if not ok[prim]:
            continue
        # Candidate attack starts are in the final test segment, after calibration.
        okp = np.isfinite(D[prim])
        starts = [s for s in range(cal_end + PRE, T - L - 1) if okp[s:s + L].all()]
        Kt = KT.get(tname, K)
        if len(starts) < Kt:
            continue
        for mag in tg['mags']:
            S = rng.choice(starts, Kt, replace=False)
            # every subset of devices that contains the target device; () is the no-attack control
            subsets = [()] + [(0,) + c for r in range(len(devs)) for c in itertools.combinations(range(1, len(devs)), r)]
            for profile, profile_shape in PROFILES.items():
                delta = mag * profile_shape
                for sub in subsets:
                    kk = len(sub)
                    chans = [c for i, d in enumerate(devs) if i in sub for c in d[1]]
                    for s in S:
                        sl = slice(s - PRE, s + L)
                        Dseg = {k: v[sl].copy() for k, v in D.items()}
                        Xseg = np.column_stack([Xf[c][sl].copy() for c in IFC])
                        if kk and not within_channel_ranges(Dseg, chans, delta):
                            RANGE_REJECTIONS.append(dict(case=case, target=tname, mag=mag,
                                                         profile=profile, k=kk, devices='+'.join(devs[i][0] for i in sub)))
                            continue
                        for c in chans:
                            Dseg[c][PRE:] += delta
                            if c in IFC:
                                Xseg[PRE:, IFC.index(c)] += delta
                        sc = inv_scores(Dseg, par)
                        fired = [nm for nm, v in sc.items() if (v[PRE:] > thr[nm]).any()]
                        fired_target = [nm for nm, v in sc.items()
                                        if nm in target_thr and (v[PRE:] > target_thr[nm]).any()]
                        fired_target_no_c4 = [nm for nm, v in sc.items()
                                              if nm in target_thr_no_c4 and (v[PRE:] > target_thr_no_c4[nm]).any()]
                        iso_seg_score = ew(-iso.score_samples(feats(Xseg)))[PRE:]
                        det_iso = (iso_seg_score > thr_iso).any()
                        det_iso_target = (iso_seg_score > iso_target_thr).any()
                        red_target_score = max(
                            (np.max(sc[nm][PRE:]) - target_thr[nm]) / max(abs(target_thr[nm]), 1.0)
                            for nm in target_thr)
                        red_target_score_no_c4 = max(
                            (np.max(sc[nm][PRE:]) - target_thr_no_c4[nm]) / max(abs(target_thr_no_c4[nm]), 1.0)
                            for nm in target_thr_no_c4)
                        iso_target_score = np.max(iso_seg_score) - iso_target_thr
                        rows.append(dict(case=case, target=tname, mag=mag, profile=profile, k=kk, kmax=len(devs),
                                         devices='+'.join(devs[i][0] for i in sub),
                                         cuff=bool(np.isfinite(D['NMBP'][s:s + L]).any()),
                                         start_index=s, cal_windows=len(cal_starts),
                                         active_invariants=','.join(par),
                                         det_red=bool(fired), det_red_target=bool(fired_target),
                                         det_red_target_no_c4=bool(fired_target_no_c4),
                                         det_iso=bool(det_iso), det_iso_target=bool(det_iso_target),
                                         red_target_score=red_target_score, red_target_score_no_c4=red_target_score_no_c4,
                                         iso_target_score=iso_target_score,
                                         fired=','.join(fired), fired_target=','.join(fired_target)))
    return rows


def write_case_bootstrap_summaries():
    """Summarize one metric row per case, then bootstrap cases rather than windows."""
    results = pd.read_csv('pilot6_results.csv')
    costs = pd.read_csv('pilot6_cost.csv')
    case_rows = []
    for (case, target), group in results.groupby(['case', 'target']):
        clean = group[group.k == 0]
        single = group[group.k == 1]
        attacked = group[group.k > 0]
        y = np.r_[np.zeros(len(clean)), np.ones(len(attacked))]
        red_auc = roc_auc_score(y, np.r_[clean.red_target_score, attacked.red_target_score])
        no_c4_auc = roc_auc_score(y, np.r_[clean.red_target_score_no_c4, attacked.red_target_score_no_c4])
        iso_auc = roc_auc_score(y, np.r_[clean.iso_target_score, attacked.iso_target_score])
        cc = costs[(costs.case == case) & (costs.target == target)]
        case_rows.append(dict(
            case=case, target=target,
            red_clean_fpr=clean.det_red_target.mean(),
            red_no_c4_clean_fpr=clean.det_red_target_no_c4.mean(),
            iforest_clean_fpr=clean.det_iso_target.mean(),
            red_single_tpr=single.det_red_target.mean(),
            red_no_c4_single_tpr=single.det_red_target_no_c4.mean(),
            iforest_single_tpr=single.det_iso_target.mean(),
            red_auc_all_attacks=red_auc, red_no_c4_auc_all_attacks=no_c4_auc,
            iforest_auc_all_attacks=iso_auc,
            evasion_config_fraction=cc.cost.notna().mean(),
            attack_cost_mean_when_found=cc.cost.mean()))
    case_df = pd.DataFrame(case_rows)
    case_df['c4_tpr_gain'] = case_df.red_single_tpr - case_df.red_no_c4_single_tpr
    case_df['c4_fpr_change'] = case_df.red_clean_fpr - case_df.red_no_c4_clean_fpr
    case_df['c4_auc_gain'] = case_df.red_auc_all_attacks - case_df.red_no_c4_auc_all_attacks
    case_df.to_csv('pilot6_case_metrics.csv', index=False)

    metrics = ['red_clean_fpr', 'red_no_c4_clean_fpr', 'iforest_clean_fpr',
               'red_single_tpr', 'red_no_c4_single_tpr', 'iforest_single_tpr',
               'red_auc_all_attacks', 'red_no_c4_auc_all_attacks', 'iforest_auc_all_attacks',
               'c4_tpr_gain', 'c4_fpr_change', 'c4_auc_gain',
               'evasion_config_fraction', 'attack_cost_mean_when_found']
    rng_boot = np.random.default_rng(20261022)
    summary_rows = []
    for target, group in case_df.groupby('target'):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            samples = rng_boot.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            summary_rows.append(dict(
                target=target, metric=metric, n_cases=len(values), mean=float(values.mean()),
                ci95_low=float(np.quantile(samples, 0.025)),
                ci95_high=float(np.quantile(samples, 0.975))))
    boot = pd.DataFrame(summary_rows)
    boot.to_csv('pilot6_cluster_bootstrap.csv', index=False)
    labels = {
        'red_clean_fpr': 'redundancy_test_fpr',
        'red_no_c4_clean_fpr': 'redundancy_no_c4_test_fpr',
        'iforest_clean_fpr': 'iforest_test_fpr',
        'red_single_tpr': 'redundancy_single_device_tpr',
        'red_no_c4_single_tpr': 'redundancy_no_c4_single_device_tpr',
        'iforest_single_tpr': 'iforest_single_device_tpr',
        'red_auc_all_attacks': 'redundancy_case_mean_auc',
        'red_no_c4_auc_all_attacks': 'redundancy_no_c4_case_mean_auc',
        'iforest_auc_all_attacks': 'iforest_case_mean_auc',
        'c4_tpr_gain': 'c4_single_device_tpr_gain',
        'c4_fpr_change': 'c4_test_fpr_change',
        'c4_auc_gain': 'c4_case_mean_auc_gain',
        'evasion_config_fraction': 'evasive_config_fraction',
        'attack_cost_mean_when_found': 'attack_cost_mean_conditional',
    }
    headline = []
    for target, group in case_df.groupby('target'):
        row = {'target': target, 'n_cases': int(group.case.nunique())}
        for metric, label in labels.items():
            values = boot[(boot.target == target) & (boot.metric == metric)]
            if len(values):
                v = values.iloc[0]
                row[label] = v['mean']
                row[f'{label}_ci95_low'] = v['ci95_low']
                row[f'{label}_ci95_high'] = v['ci95_high']
        headline.append(row)
    headline_df = pd.DataFrame(headline)
    headline_df.to_csv('pilot6_summary.csv', index=False)
    ablation = headline_df[['target', 'n_cases',
                            'redundancy_single_device_tpr',
                            'redundancy_single_device_tpr_ci95_low',
                            'redundancy_single_device_tpr_ci95_high',
                            'redundancy_no_c4_single_device_tpr',
                            'redundancy_no_c4_single_device_tpr_ci95_low',
                            'redundancy_no_c4_single_device_tpr_ci95_high',
                            'redundancy_test_fpr', 'redundancy_no_c4_test_fpr',
                            'redundancy_case_mean_auc', 'redundancy_no_c4_case_mean_auc',
                            'c4_single_device_tpr_gain', 'c4_single_device_tpr_gain_ci95_low',
                            'c4_single_device_tpr_gain_ci95_high', 'c4_test_fpr_change',
                            'c4_test_fpr_change_ci95_low', 'c4_test_fpr_change_ci95_high',
                            'c4_case_mean_auc_gain', 'c4_case_mean_auc_gain_ci95_low',
                            'c4_case_mean_auc_gain_ci95_high']].copy()
    ablation.to_csv('pilot6_ablation_summary.csv', index=False)


if __name__ == '__main__':
    allrows = []
    for f in sorted(glob.glob(os.path.join(DATA, 'case*.csv'))):
        if f.endswith('_wave.csv'):
            continue
        t0 = time.time(); r = run_case(f); allrows += r
        print(f'{os.path.basename(f)}: {len(r)} attacks, {time.time() - t0:.0f}s', flush=True)
    d = pd.DataFrame(allrows); d.to_csv('pilot6_results.csv', index=False)
    pd.DataFrame(RANGE_REJECTIONS).to_csv('pilot6_range_rejections.csv', index=False)
    audit = (d[d.k == 0].groupby(['case', 'target', 'profile'])
             .agg(calibration_windows=('cal_windows', 'first'),
                  active_invariants=('active_invariants', 'first'),
                  global_detector_test_fpr=('det_red', 'mean'),
                  target_redundancy_test_fpr=('det_red_target', 'mean'),
                  redundancy_no_c4_test_fpr=('det_red_target_no_c4', 'mean'),
                  target_iforest_test_fpr=('det_iso_target', 'mean'))
             .reset_index())
    audit.to_csv('pilot6_control_audit.csv', index=False)
    # Test ROC operating points: thresholds are selected from clean test controls
    # solely to compare detection at the same realized test-window FPR. These
    # are descriptive ROC results, not deployable thresholds or an independent test.
    matched_rows = []
    controls = d[d.k == 0]
    attacks = d[d.k > 0]
    for (target, mag, profile), clean in controls.groupby(['target', 'mag', 'profile']):
        for target_fpr in (0.05, 0.10):
            red_cut, red_tie_p, red_fpr = matched_cut(clean.red_target_score, target_fpr)
            iso_cut, iso_tie_p, iso_fpr = matched_cut(clean.iso_target_score, target_fpr)
            sub = attacks[(attacks.target == target) & (attacks.mag == mag) & (attacks.profile == profile)]
            for (case, k, devices), attacked in sub.groupby(['case', 'k', 'devices']):
                matched_rows.append(dict(
                    case=case, target=target, mag=mag, profile=profile, target_fpr=target_fpr,
                    k=k, devices=devices, n_clean=len(clean), n_attack=len(attacked),
                    red_clean_fpr=red_fpr,
                    red_detection=matched_detection(attacked.red_target_score, red_cut, red_tie_p),
                    iforest_clean_fpr=iso_fpr,
                    iforest_detection=matched_detection(attacked.iso_target_score, iso_cut, iso_tie_p)))
    matched = pd.DataFrame(matched_rows)
    matched.to_csv('pilot6_matched_fpr.csv', index=False)
    # At the 10% test-FPR operating point, define evasion as detection no more
    # than the realized clean FPR plus 10 percentage points. This uses the test
    # controls and must be presented as a descriptive ROC-based cost only.
    m10 = matched[matched.target_fpr == 0.10].copy()
    m10['evades'] = m10.red_detection <= m10.red_clean_fpr + 0.10
    ev = m10[m10.evades].sort_values(['case', 'target', 'mag', 'profile', 'k', 'red_detection'])
    best = ev.groupby(['case', 'target', 'mag', 'profile']).first().reset_index()[
        ['case', 'target', 'mag', 'profile', 'k', 'devices']].rename(columns={'k': 'cost', 'devices': 'cheapest_set'})
    kmax = m10.groupby(['case', 'target', 'mag', 'profile']).k.max().rename('kmax').reset_index()
    matched_cost = kmax.merge(best, on=['case', 'target', 'mag', 'profile'], how='left')
    matched_cost.to_csv('pilot6_cost_matched_fpr.csv', index=False)
    matched_auc = []
    for (target, mag, profile), clean in controls.groupby(['target', 'mag', 'profile']):
        sub = attacks[(attacks.target == target) & (attacks.mag == mag) & (attacks.profile == profile)]
        for detector, col in [('redundancy', 'red_target_score'), ('Isolation Forest', 'iso_target_score')]:
            scores = np.r_[clean[col].to_numpy(), sub[col].to_numpy()]
            labels = np.r_[np.zeros(len(clean)), np.ones(len(sub))]
            matched_auc.append(dict(target=target, mag=mag, profile=profile, detector=detector,
                                    test_roc_auc=roc_auc_score(labels, scores)))
    pd.DataFrame(matched_auc).to_csv('pilot6_test_roc_auc.csv', index=False)
    print('\nDescriptive test ROC AUC by target and magnitude:')
    print(pd.DataFrame(matched_auc).pivot(index=['target', 'mag', 'profile'], columns='detector', values='test_roc_auc').round(3).to_string())
    print('\nMatched-FPR detections (test controls set the ROC thresholds; descriptive only):')
    matched_summary = matched.groupby(['target', 'mag', 'profile', 'target_fpr']).agg(
        red_clean_fpr=('red_clean_fpr', 'first'), red_detection=('red_detection', 'mean'),
        iforest_clean_fpr=('iforest_clean_fpr', 'first'), iforest_detection=('iforest_detection', 'mean'))
    print(matched_summary.round(3).to_string())
    print('\nMatched-FPR attack-cost means at 10% operating point:')
    print(matched_cost.groupby(['target', 'mag', 'profile']).cost.mean().round(2).to_string())
    if RANGE_REJECTIONS:
        print(f'\nRejected out-of-range attack windows: {len(RANGE_REJECTIONS)}')
    global_control = d[d.k == 0].groupby(['target', 'profile']).det_red.mean()
    control = (d[d.k == 0].groupby(['case', 'target', 'profile']).det_red_target.mean()
               .rename('control_rate').reset_index())
    base = control.groupby(['target', 'profile']).control_rate.mean()
    base_iso_target = d[d.k == 0].groupby(['target', 'profile']).det_iso_target.mean()
    print('\nNo-attack control: overall detector false-alarm rate by target:'); print(global_control.round(3).to_string())
    print(f'\nNo-attack control: target-specific redundancy FPR (calibration target {TARGET_FPR:.0%}):'); print(base.round(3).to_string())
    print('\nNo-attack control: target-specific Isolation Forest FPR:'); print(base_iso_target.round(3).to_string())
    d = d[d.k > 0]
    g = d.groupby(['target', 'mag', 'profile', 'k', 'devices']).agg(
        n=('det_red', 'size'), global_detector=('det_red', 'mean'),
        redundancy_matched=('det_red_target', 'mean'), iforest_global=('det_iso', 'mean'),
        iforest_matched=('det_iso_target', 'mean'))
    print('\nFraction of attacks detected, per set of compromised devices (all cases):'); print(g.round(2).to_string())
    # attack cost: smallest device set whose detection rate (per case) is <= control false-alarm rate + 10 points
    c = d.groupby(['case', 'target', 'profile', 'mag', 'k', 'devices']).det_red_target.mean().reset_index()
    c = c.merge(control, on=['case', 'target', 'profile'])
    # Define evasion relative to the clean false-alarm rate for the same patient.
    # This prevents high-false-alarm cases from being silently compared with a
    # pooled threshold, while keeping the reported clean rate visible.
    ev = c[c.det_red_target <= c.control_rate + 0.10].sort_values(['case', 'target', 'profile', 'mag', 'k', 'det_red_target'])
    best = ev.groupby(['case', 'target', 'profile', 'mag']).first().reset_index()[['case', 'target', 'profile', 'mag', 'k', 'devices']].rename(columns={'k': 'cost', 'devices': 'cheapest_set'})
    kmax = d.groupby(['case', 'target', 'profile', 'mag']).kmax.first().reset_index()
    cost = kmax.merge(best, how='left')
    cost.to_csv('pilot6_cost.csv', index=False)
    # Headline summaries and confidence intervals are clustered by case, not window.
    write_case_bootstrap_summaries()
    print('\nIndependent-calibration headline summary:')
    print(pd.read_csv('pilot6_summary.csv').round(3).to_string(index=False))
    print('\nAttack cost = cheapest device set with target-relevant detection no more than that case\'s clean FPR + 10 points; NaN = nothing evades:')
    print(cost.groupby(['target', 'mag', 'profile']).cost.agg(['mean', 'min', 'max', lambda x: x.isna().sum()]).rename(columns={'<lambda_0>': 'never_evaded'}).round(2).to_string())
    print('\nCheapest evading set, count over cases:')
    print(cost.groupby(['target', 'mag', 'profile']).cheapest_set.value_counts(dropna=False).to_string())
