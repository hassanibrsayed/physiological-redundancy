"""Diagnostic for the RR result under joint thresholds (evasion found in only 0-25% of goals, detection 1.0 at
every device count). Question: are the RR invariants protecting against consistent multi-device attacks, or do
the flat, integer-valued RR channels make every non-integer or non-exact perturbation trip the detector?

For each case it prints
  1. whether RR, ventilator RR and anaesthesia-machine RR are integer-valued and how much they vary,
  2. the joint and Bonferroni thresholds of every RR invariant and how often the clean calibration maxima are 0,
  3. a test with all three RR devices shifted by the SAME offset (ramp of 20 s, 0.5 SD), once with continuous
     values and once rounded to whole breaths/min: share of windows with an alarm, and which invariants fire,
     for the joint thresholds with and without the C4 rate checks, and for a no-attack control.
Run:  .\\.venv\\Scripts\\python.exe diag_rr.py > diag_rr.log     Writes nothing else."""
import os
os.environ['JOINT'] = '1'
import numpy as np
import pilot6 as p6
import pilot11_redteam as r

PRE, L = p6.PRE, p6.L
CASES = [10, 16, 17, 19, 29, 34, 55, 60]


def fired(D, par, thr, s, shifts):
    sl = slice(s - PRE, s + L)
    Dseg = {k: v[sl].copy() for k, v in D.items()}
    for c, dl in shifts.items():
        Dseg[c][PRE:] += dl
    sc = p6.inv_scores(Dseg, {n: par[n] for n in thr})
    return [n for n in thr if np.max(sc[n][PRE:]) > thr[n]]


for cid in CASES:
    D = p6.load(f'../vitaldb_cases_ext/case{cid}.csv')
    par, mx, ok, te, ce = r.calibrate(D)
    print(f'\n===== case {cid} =====')
    for ch in ['RR', 'VRR', 'RRP']:
        v = D[ch][:te]; v = v[np.isfinite(v)]
        if len(v):
            print(f'  {ch}: n={len(v)} integer-valued={np.mean(np.abs(v - np.round(v)) < 1e-6):.2f} '
                  f'SD={np.std(v):.2f} distinct values={len(np.unique(v))}')
    r.JOINT = True;  thj = r.thresholds(par, mx, 'RR', True)
    r.JOINT = False; thb = r.thresholds(par, mx, 'RR', True)
    if thj is None:
        print('  no RR invariants'); continue
    print('  threshold (joint / Bonferroni) and clean calibration maxima (share of windows equal to 0, median, 90th pct):')
    for n in thj:
        m = mx[n][np.isfinite(mx[n])]
        print(f'    {n:7s} {thj[n]:7.3f} / {thb[n]:7.3f}   zero share {np.mean(m <= 1e-9):.2f}  median {np.median(m):.3f}  p90 {np.quantile(m, 0.9):.3f}')
    r.JOINT = True
    thj_no = r.thresholds(par, mx, 'RR', False)
    devs = p6.TARGETS['RR']['devs']
    devs = [d for i, d in enumerate(devs) if i == 0 or all(ok[c] for c in d[1])]
    sd = float(np.nanstd(D['RR'][:te]))
    okp = np.isfinite(D['RR']); T = len(D['RR'])
    cand = [s for s in range(ce + PRE, T - L - 1) if okp[s:s + L].all()]
    if len(cand) < 10:
        print('  too few candidate windows'); continue
    starts = [cand[i] for i in np.linspace(0, len(cand) - 1, 25).astype(int)]
    t = np.arange(1, L + 1, dtype=float)
    for label, quant in [('continuous offset', False), ('whole-breath offsets', True)]:
        off = 0.5 * sd if not quant else max(1.0, round(0.5 * sd))
        d = off * np.clip(t / 20, 0, 1)
        if quant:
            d = np.round(d)
        shifts = {c: d for _, cs in devs for c in cs}
        for dname, thr in [('with C4', thj), ('without C4', thj_no)]:
            hits = [fired(D, par, thr, s, shifts) for s in starts]
            ctrl = [fired(D, par, thr, s, {}) for s in starts]
            names = {}
            for h in hits:
                for n in h:
                    names[n] = names.get(n, 0) + 1
            print(f'  {label:22s} {dname:10s} (offset {off:.2f}, all {len(devs)} devices): alarm in {np.mean([bool(h) for h in hits]):.2f} of windows '
                  f'(control {np.mean([bool(h) for h in ctrl]):.2f}); fired: {names}')
