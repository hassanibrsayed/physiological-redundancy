# Replicate the joint-threshold pipeline on the 12 sample cases that were not used for any design decision.
# Needs internet only for step 1 (arterial waveform from VitalDB). Run from this folder in PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\run_replication.ps1
# About 25 minutes. Outputs are pilot14_* (grid, baselines, combined detector) and pilot15_* (red team);
# nothing from the 8-case runs is overwritten. Final comparison: compare_replication.log
$py = ".\.venv\Scripts\python.exe"
$env:DATA_DIR = "cgptcode_expanded\vitaldb_cases_ext"

# 1. pulse rate from the arterial waveform for the 12 cases (case 56 has no waveform and is skipped)
$env:WAVE_CASES = "20,22,27,28,4,43,50,53,56,64,65,75"
$env:TAG = "rep"
& $py download_wave.py > run14_wave.log 2>&1
Remove-Item Env:WAVE_CASES, Env:TAG

# 2. attacks in SD units with joint thresholds (pilot14.py is pilot9.py with the output prefix renamed)
$env:JOINT = "1"
(Get-Content pilot9.py -Raw) -replace 'pilot9_', 'pilot14_' | Set-Content pilot14.py -Encoding utf8
& $py pilot14.py > run14.log 2>&1

# 3. learned baselines and combined detector on exactly those windows
& $py pilot7_baselines.py pilot14 > run14_base.log 2>&1
& $py -W ignore pilot8_combined.py pilot14 > run14_comb.log 2>&1

# 4. red team
$env:TAG = "pilot15"
& $py pilot11_redteam.py > run15.log 2>&1

# 5. comparison with the 8 design cases
Remove-Item Env:JOINT, Env:TAG, Env:DATA_DIR
& $py -W ignore compare_replication.py | Tee-Object -FilePath compare_replication.log
