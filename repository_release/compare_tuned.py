"""Compare the untuned learned baselines (Pilot 7: pilot12 / pilot14) with the tuned ones (Pilot 16:
pilot12tuned / pilot14tuned) after pilot8_combined.py has been run on the tuned score files.

Run:  .\\.venv\\Scripts\\python.exe compare_tuned.py        Reads *_comb_detection.csv and *_comb_cost.csv; writes nothing."""
import pandas as pd

pd.set_option('display.width', 220)
SETS = {'8 design cases': ('pilot12', 'pilot12tuned'), '12 replication cases': ('pilot14', 'pilot14tuned')}
DETS = ['Redundancy', 'Learned', 'IF (all ch)', 'Autoencoder', 'LSTM', 'Split OR (5%+5%)']


def det_k(prefix):
    d = pd.read_csv(f'{prefix}_comb_detection.csv')
    return d[d.detector.isin(DETS)].groupby(['target', 'k', 'detector']).detection.mean().unstack('detector')[DETS].round(2)


def cost(prefix):
    c = pd.read_csv(f'{prefix}_comb_cost.csv')
    ad = c.groupby(['detector', 'case', 'target', 'mag']).cost.min().reset_index()
    t = ad[ad.detector.isin(DETS)].groupby(['target', 'detector']).agg(
        evasion=('cost', lambda x: x.notna().mean()), cost=('cost', 'mean')).round(2).unstack('detector')
    return t.reindex(columns=pd.MultiIndex.from_product([['evasion', 'cost'], DETS]))


for name, (old, new) in SETS.items():
    print(f'\n=== {name}: detection at a matched 10% false-alarm rate, by number of compromised sources ===')
    print('-- untuned baselines'); print(det_k(old).to_string())
    print('-- tuned baselines'); print(det_k(new).to_string())
    print(f'\n=== {name}: grid attack cost (evasion found / mean cost), attacker free to choose the profile ===')
    print('-- untuned'); print(cost(old).to_string())
    print('-- tuned'); print(cost(new).to_string())
