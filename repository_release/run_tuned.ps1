# Tune the learned baselines without touching test windows, re-score the Pilot 12 and Pilot 14 attack windows,
# rerun the combined-detector analysis on them and compare with the untuned baselines.
# Needs pilot12_base_scores.csv and pilot14_base_scores.csv (from run_joint.ps1 and run_replication.ps1).
# Run from this folder in PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\run_tuned.ps1
# About 40 minutes. Outputs: pilot16_*.csv, pilot12tuned_*, pilot14tuned_*, logs run16*.log, compare_tuned.log.
$py = ".\.venv\Scripts\python.exe"
& $py pilot16_tuned_baselines.py > run16.log 2>&1
& $py -W ignore pilot8_combined.py pilot12tuned > run16_comb12.log 2>&1
& $py -W ignore pilot8_combined.py pilot14tuned > run16_comb14.log 2>&1
& $py -W ignore compare_tuned.py | Tee-Object -FilePath compare_tuned.log
