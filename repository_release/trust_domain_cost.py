"""Attack cost counted in trust domains instead of measurement sources.

Cost in sources assumes every measurement source can be compromised on its own. In the recordings most channels
come from one patient monitor (the Solar8000 tracks of VitalDB); the anaesthesia machine (Primus) and the second
arterial monitor (EV1000) are separate devices. An attacker who controls a device controls every channel it
reports, so the number of trust domains the attacker must control can be smaller than the number of sources.

Method (no new attack search is needed). The red team already searched every subset of sources that contains the
target source and recorded which subsets succeeded. For each goal we map every successful subset to the set of
domains its sources belong to and take the smallest number of domains. Two assignments are reported:
  A  conservative: the ventilator rate (VitalDB track Solar8000/VENT_RR) belongs to the patient monitor
  B  the ventilator rate belongs to the anaesthesia machine (the ventilator is part of the machine)
Domains: MON = patient monitor (ECG, pleth, arterial line and its waveform, cuff, monitor capnograph),
         ANA = anaesthesia machine (its capnograph; ventilator rate in B), EV = EV1000 second arterial monitor.
Arterial waveform pulse rate is assigned to MON because the waveform comes from the same transducer and monitor.

Run:  .\\.venv\\Scripts\\python.exe trust_domain_cost.py
Reads pilot13_redteam.csv (8 design cases) and pilot15_redteam.csv (12 replication cases; joint thresholds, timing-free
mode). Writes trust_domain_cost.csv and journey_figures/09_trust_domain_cost.png."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOM_A = {'ECG': 'MON', 'PLETH': 'MON', 'ART_WAVE': 'MON', 'ART': 'MON', 'NIBP': 'MON', 'EV1000': 'EV',
         'MON_CAPNO': 'MON', 'MACHINE_CAPNO': 'ANA', 'VENT': 'MON'}
DOM_B = {**DOM_A, 'VENT': 'ANA'}
EV_CASES = {17, 19, 29, 34, 55, 60}                      # design cases with an EV1000 track
KEYS = ['case', 'target', 'detector', 'mag', 'sign', 'mode']
TN = {'HR': 'Heart rate', 'BP': 'Arterial MBP', 'ETCO2': 'EtCO2', 'RR': 'Respiratory rate'}


def cost_table(path, label):
    d = pd.read_csv(path)
    d = d[d['mode'] == 'timing-free']
    s = d[d.success].copy()
    for name, dom in (('A', DOM_A), ('B', DOM_B)):
        s[f'dom_{name}'] = s.devices.apply(lambda x: len({dom[y] for y in x.split('+')}))
    g = s.groupby(KEYS).agg(sources=('k', 'min'), domains_A=('dom_A', 'min'), domains_B=('dom_B', 'min'))
    cells = d[KEYS].drop_duplicates().set_index(KEYS).join(g).reset_index()
    cells['set'] = label
    cells['evm'] = cells.case.isin(EV_CASES)
    return cells


cells = pd.concat([cost_table('pilot13_redteam.csv', 'design (8)'), cost_table('pilot15_redteam.csv', 'replication (12)')])
cells.to_csv('trust_domain_cost.csv', index=False)


def summary(c):
    return c.groupby(['detector', 'target']).agg(
        goals=('sources', 'size'), evasion_found=('sources', lambda x: x.notna().mean()),
        sources=('sources', 'mean'), domains_A=('domains_A', 'mean'), domains_B=('domains_B', 'mean')).round(2)


pd.set_option('display.width', 200)
for name, c in (('Design cases (8)', cells[cells.set == 'design (8)']),
                ('Replication cases (12)', cells[cells.set == 'replication (12)']),
                ('Pooled over the 20 cases (goals weighted equally)', cells)):
    print(f'\n{name}: mean cost in sources and in trust domains (A: ventilator rate on the monitor; B: on the anaesthesia machine)')
    print(summary(c).to_string())
bp = cells[(cells.target == 'BP') & (cells.detector == 'without C4')]
print('\nArterial pressure, without C4, split by presence of EV1000:')
print(bp.groupby('evm').agg(goals=('sources', 'size'), evasion_found=('sources', lambda x: x.notna().mean()),
                            sources=('sources', 'mean'), domains_A=('domains_A', 'mean')).round(2).to_string())

# figure: sources versus domains, no C4, pooled
nc = summary(cells).loc['without C4'].reindex(['HR', 'BP', 'ETCO2', 'RR'])
fig, ax = plt.subplots(figsize=(7.5, 4))
x = np.arange(len(nc)); w = 0.27
ax.bar(x - w, nc.sources, w, label='Sources (independent compromise)', color='#1f6fb4')
ax.bar(x, nc.domains_A, w, label='Trust domains, ventilator rate on the monitor', color='#e08a8a')
ax.bar(x + w, nc.domains_B, w, label='Trust domains, ventilator rate on the anaesthesia machine', color='#bcd4e6')
ax.set_xticks(x); ax.set_xticklabels([TN[t] for t in nc.index]); ax.set_ylabel('Attack cost (mean)')
ax.set_ylim(0, 3.4); ax.grid(axis='y', alpha=0.25); ax.legend(fontsize=7, frameon=False, loc='upper left')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
os.makedirs('journey_figures', exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join('journey_figures', '09_trust_domain_cost.png'), dpi=160)
print('\nwrote trust_domain_cost.csv and journey_figures/09_trust_domain_cost.png')
