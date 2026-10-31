"""Download the redundant-device tracks needed for R1-R5 / C1-C4 from VitalDB.
Run on the user's own computer (VitalDB is blocked from Claude's cloud workspace).

    python download_tracks.py            # numeric tracks only (small)
    python download_tracks.py --wave     # also the 500 Hz arterial waveform, downsampled

Output: ../vitaldb_cases_ext/case<ID>.csv (one row per INTERVAL seconds, NaN where a
device did not report) and download_report.csv (per case x track: found, % non-NaN).
Tracks that a case does not have are skipped and reported, not treated as errors."""
import argparse, os, warnings
import numpy as np, pandas as pd
import vitaldb

CASEIDS = [4, 20, 22, 27, 28, 43, 50, 53, 56, 64, 65, 75]
INTERVAL = 1.0  # seconds between samples; Solar8000 numerics update about every 2 s
OUT = os.path.join('vitaldb_cases_ext')

# redundancy group -> tracks (names to be confirmed against get_track_names; missing ones are skipped)
TRACKS = {
    'R1/R2 heart rate': ['Solar8000/HR', 'Solar8000/PLETH_HR', 'CardioQ/HR'],
    'R2/R3/C1-C3 arterial': ['Solar8000/ART_SBP', 'Solar8000/ART_MBP', 'Solar8000/ART_DBP', 'EV1000/ART_MBP'],
    'R3 cuff': ['Solar8000/NIBP_SBP', 'Solar8000/NIBP_MBP', 'Solar8000/NIBP_DBP'],
    'extra BP sources': ['Solar8000/FEM_MBP', 'Solar8000/PA_MBP'],
    'R4 EtCO2': ['Solar8000/ETCO2', 'Primus/ETCO2'],
    'R5 respiration': ['Solar8000/RR_CO2', 'Primus/RR_CO2', 'Solar8000/VENT_RR', 'Primus/SET_RR_IMV'],
    'context': ['Solar8000/PLETH_SPO2', 'Solar8000/BT'],
}
ALL = [t for g in TRACKS.values() for t in g]


def main(wave):
    os.makedirs(OUT, exist_ok=True)
    warnings.simplefilter('ignore', DeprecationWarning)
    names = vitaldb.get_track_names(CASEIDS).set_index('caseid')['tnames'].to_dict()
    report = []
    for cid in CASEIDS:
        have = set(names.get(cid, []))
        want = [t for t in ALL if t in have]
        if wave and 'SNUADC/ART' in have:
            want.append('SNUADC/ART')
        for t in ALL:
            if t not in have:
                report.append(dict(caseid=cid, track=t, found=False, pct_valid=0.0))
        if not want:
            print(f'case {cid}: nothing to download'); continue
        iv = INTERVAL if not wave else 1 / 100  # 100 Hz grid when the waveform is requested
        arr = vitaldb.load_case(cid, want, iv)
        df = pd.DataFrame(arr, columns=want)
        df.insert(0, 'time_s', np.arange(len(df)) * iv)
        if wave:  # keep numerics on the same grid; analysis code can resample
            df.to_csv(os.path.join(OUT, f'case{cid}_wave.csv'), index=False)
        else:
            df.to_csv(os.path.join(OUT, f'case{cid}.csv'), index=False)
        for t in want:
            report.append(dict(caseid=cid, track=t, found=True, pct_valid=round(100 * df[t].notna().mean(), 1)))
        print(f'case {cid}: {len(df)} rows, {len(want)}/{len(ALL)} tracks')
    pd.DataFrame(report).sort_values(['caseid', 'track']).to_csv('download_report.csv', index=False)
    print('wrote download_report.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--wave', action='store_true')
    main(ap.parse_args().wave)
