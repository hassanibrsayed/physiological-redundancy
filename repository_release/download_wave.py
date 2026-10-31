"""Download the arterial waveform (SNUADC/ART) for the 8 pilot cases and derive a pulse rate from it (R2).

Run on the user's computer (VitalDB is blocked in Claude's cloud workspace):
    .\\.venv\\Scripts\\python.exe download_wave.py

For each case the waveform is loaded at 100 Hz, systolic peaks are detected, and a causal pulse rate is
computed on a 1 s grid as 60 / median inter-beat interval over the trailing 6 s (NaN with fewer than 3 valid
intervals). Output: ../vitaldb_cases_ext/artpr/case<ID>.csv with columns time_s, ART_PR. The raw waveform is
not stored. wave_report.csv gives the fraction of valid seconds and the median |ECG HR - ART_PR| per case."""
import os
import numpy as np, pandas as pd
import vitaldb
from scipy.signal import find_peaks

# Defaults are the 8 pilot cases. For the replication cases set DATA_DIR (folder with case<ID>.csv),
# WAVE_CASES (comma-separated ids) and TAG (suffix of the report file); see run_replication.ps1.
CASEIDS = [int(x) for x in os.environ['WAVE_CASES'].split(',')] if os.environ.get('WAVE_CASES') else [55, 17, 19, 34, 60, 29, 10, 16]
FS = 100
WIN = 6          # seconds of trailing history for the pulse rate
OUT = os.environ.get('DATA_DIR', os.path.join('..', 'vitaldb_cases_ext'))
REPORT = f"wave_report_{os.environ['TAG']}.csv" if os.environ.get('TAG') else 'wave_report.csv'


def pulse_rate_from_wave(w):
    x = np.nan_to_num(w, nan=0.0)
    # systolic peaks: at least 0.27 s apart (about 220 bpm), above 30 mmHg, prominent against the local trough
    pk, _ = find_peaks(x, distance=int(0.27 * FS), height=30, prominence=15)
    t = pk / FS
    ibi = np.diff(t)
    good = (ibi > 60 / 220) & (ibi < 60 / 25)
    bt, bi = t[1:][good], ibi[good]
    n = int(len(w) / FS)
    out = np.full(n, np.nan)
    lo = 0
    for s in range(n):
        while lo < len(bt) and bt[lo] < s + 1 - WIN:
            lo += 1
        hi = np.searchsorted(bt, s + 1, side='left')
        if hi - lo >= 3:
            out[s] = 60 / np.median(bi[lo:hi])
    return out


if __name__ == '__main__':
    rep = []
    os.makedirs(os.path.join(OUT, 'artpr'), exist_ok=True)
    for cid in CASEIDS:
        try:
            w = vitaldb.load_case(cid, ['SNUADC/ART'], 1 / FS)[:, 0]
        except Exception as e:                     # e.g. the case has no arterial waveform
            print(f'case {cid}: no arterial waveform ({e}); R2 will be skipped for this case', flush=True)
            continue
        if not np.isfinite(w).any():
            print(f'case {cid}: arterial waveform is empty; R2 will be skipped for this case', flush=True)
            continue
        pr = pulse_rate_from_wave(w)
        df = pd.DataFrame({'time_s': np.arange(len(pr), dtype=float), 'ART_PR': pr})
        df.to_csv(os.path.join(OUT, 'artpr', f'case{cid}.csv'), index=False)
        ref = pd.read_csv(os.path.join(OUT, f'case{cid}.csv'))
        hr = ref['Solar8000/HR'].reindex(range(len(pr))).values
        diff = np.abs(hr - pr)
        rep.append(dict(caseid=cid, seconds=len(pr), pct_valid=round(100 * np.isfinite(pr).mean(), 1),
                        median_abs_diff_vs_ecg=round(float(np.nanmedian(diff)), 2),
                        p95_abs_diff_vs_ecg=round(float(np.nanpercentile(diff, 95)), 2)))
        print(rep[-1], flush=True)
    pd.DataFrame(rep).to_csv(REPORT, index=False)
