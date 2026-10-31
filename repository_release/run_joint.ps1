# Re-run Pilot 9, the learned baselines, the combined detector and the red team with joint thresholds.
# Run from this folder in PowerShell:  .\run_joint.ps1
# Takes about 15 minutes. Pilot 9/11 results are not overwritten (outputs are pilot12_* and pilot13_*).
$py = ".\.venv\Scripts\python.exe"

# 1. attacks in SD units with joint thresholds (pilot12.py is pilot9.py with the output prefix renamed)
$env:JOINT = "1"
(Get-Content pilot9.py -Raw) -replace 'pilot9_', 'pilot12_' | Set-Content pilot12.py -Encoding utf8
& $py pilot12.py > run12.log 2>&1

# 2. learned baselines and combined detector on exactly those windows
& $py pilot7_baselines.py pilot12 > run12_base.log 2>&1
& $py -W ignore pilot8_combined.py pilot12 > run12_comb.log 2>&1

# 3. red team against the joint-threshold detector
$env:TAG = "pilot13"
& $py pilot11_redteam.py > run13.log 2>&1

# 4. side-by-side comparison
Remove-Item Env:JOINT, Env:TAG
& $py -W ignore compare_joint.py | Tee-Object -FilePath compare_joint.log
