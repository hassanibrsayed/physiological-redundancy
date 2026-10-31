"""List the VitalDB tracks actually available for the 20 pilot cases, so the
R1-R5 / C1-C4 invariant plan in CLAUDE.md is checked against real track names
instead of names written from memory. Also scores each case by how many
independent BP sources it has, to help pick cases for the next step."""
import warnings
import pandas as pd
import vitaldb

CASEIDS = [4, 10, 16, 17, 19, 20, 22, 27, 28, 29, 34, 43, 50, 53, 55, 56, 60, 64, 65, 75]

with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    df = vitaldb.get_track_names(CASEIDS)  # columns: caseid, tnames (list)

BP_SOURCES = {
    'ART': 'Solar8000/ART_MBP',
    'NIBP': 'Solar8000/NIBP_MBP',
    'EV1000': 'EV1000/ART_MBP',
    'FEM': 'Solar8000/FEM_MBP',
    'PA': 'Solar8000/PA_MBP',
}
rows = []
for _, row in df.iterrows():
    cid, names = row['caseid'], set(row['tnames'])
    bp_present = [k for k, t in BP_SOURCES.items() if t in names]
    rows.append(dict(
        caseid=cid,
        n_tracks=len(names),
        n_bp_sources=len(bp_present),
        bp_sources=','.join(bp_present),
        has_ART_wave='SNUADC/ART' in names,
        has_PLETH_wave='SNUADC/PLETH' in names,
        has_ECG_V5='SNUADC/ECG_V5' in names,
        has_CardioQ_HR='CardioQ/HR' in names,
        has_NIBP_full=all(f'Solar8000/NIBP_{p}' in names for p in ['SBP', 'DBP', 'MBP']),
        has_Primus_ETCO2='Primus/ETCO2' in names,
        has_Solar_ETCO2='Solar8000/ETCO2' in names,
        has_RR_CO2=('Primus/RR_CO2' in names or 'Solar8000/RR_CO2' in names),
        has_VENT_RR='Solar8000/VENT_RR' in names,
    ))

summary = pd.DataFrame(rows).sort_values('n_bp_sources', ascending=False)
summary.to_csv('tracks_available.csv', index=False)
print(summary.to_string(index=False))
