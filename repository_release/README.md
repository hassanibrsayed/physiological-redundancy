# Cross-device physiological redundancy and the cost of false data injection in perioperative monitoring

Code, result files and logs for the manuscript *How Many Devices Must an Attacker Control? Cross-Device Physiological
Redundancy and the Cost of False Data Injection in Perioperative Monitoring*.

[Authors, affiliations and contact: to be added after review.]

## What is in this repository

| Path | Content |
|---|---|
| `download_tracks.py`, `download_wave.py`, `list_tracks.py` | Download the numeric tracks (and the arterial-waveform pulse rate) of the 8 design cases from the public VitalDB database |
| `cgptcode_expanded/download_tracks_expanded.py` | Same download for the 12 replication cases |
| `pilot6.py` | Invariants R1-R5, C1-C4, the 30/40/30 split, thresholds (Bonferroni or joint), device subsets, attack generation; also used as a module by the scripts below |
| `pilot9.py` | Grid attacks in patient-SD units, both signs, three onset profiles (`JOINT=1` selects joint thresholds, `DATA_DIR` the case folder) |
| `pilot7_baselines.py` | Isolation Forest, autoencoder and LSTM baselines on the attack windows (untuned) |
| `pilot8_combined.py` | Evidence z = -log p, matched-false-alarm comparison, fusions, grid cost |
| `pilot10_bp_twin.py` | Learned arterial-pressure twin (earlier configuration, Bonferroni, untuned baselines) |
| `pilot11_redteam.py` | Black-box red team (`TAG` sets the output prefix, `JOINT=1` joint thresholds) |
| `pilot16_tuned_baselines.py` | Tuning of the three learned baselines on the calibration segments of the design cases only; re-scoring of all 20 cases |
| `pilot17_bp_fusion.py` | Arterial-pressure fusions (grid cost) |
| `trust_domain_cost.py` | Cost in devices (trust domains) from the red-team subsets |
| `compare_joint.py`, `compare_replication.py`, `compare_tuned.py`, `diag_rr.py` | Comparisons and the respiratory-channel diagnostic |
| `bootstrap_cost_ci.py` | Case-level bootstrap intervals for the red-team cost (reads `pilot13_cost.csv` and `pilot15_cost.csv`, writes `bootstrap_cost_ci.csv`) |
| `make_figures_v4.py`, `make_journey_figures.py` | Figures |
| `pilot.py` | The first pilot (heart rate against pleth pulse rate; uses a separate set of 20 cases with 8 WUSTL-style channels, not included) |
| `run_joint.ps1`, `run_replication.ps1`, `run_tuned.ps1` | PowerShell drivers for the three main runs |
| `results/` | Result files behind the tables and figures of the paper (see below) |
| `logs/` | Console outputs of the comparison scripts and two long runs |

The recordings themselves are **not** redistributed. They come from VitalDB (Lee et al., Scientific Data 2022;9:279,
https://vitaldb.net) and must be obtained from the source under its terms of use.

## Cases

Design cases (all design decisions were made on them): 10, 16, 17, 19, 29, 34, 55, 60 (six of them, 17, 19, 29, 34, 55, 60,
have a second arterial source, EV1000). Replication cases (held back): 4, 20, 22, 27, 28, 43, 50, 53, 56, 64, 65, 75
(none has EV1000; case 56 has no arterial waveform).

## Requirements

Python 3 with the packages in `requirements.txt` (the versions we used; `torch` is the CPU build). The PowerShell scripts expect a
virtual environment at `.venv` in the repository root and are run from the repository root:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Downloading from VitalDB needs internet access.

## Reproducing the results

Every script reads and writes files in the current directory, and expects the design-case data in `..\vitaldb_cases_ext`
(one level above the repository root) and the replication data in `cgptcode_expanded\vitaldb_cases_ext`.

1. **Data.** `python download_tracks.py`, then `python download_wave.py` (design cases). For the replication cases,
   run `download_tracks_expanded.py` from inside `cgptcode_expanded`.
2. **Design cases, joint thresholds (headline results).** `.\run_joint.ps1` (about 15 minutes). It writes `pilot12_*`
   (grid, baselines, combined detector) and `pilot13_*` (red team) and prints `compare_joint.py`.
3. **Replication cases.** `.\run_replication.ps1` (about 25 minutes). It writes `pilot14_*` and `pilot15_*` and runs `compare_replication.py`.
4. **Tuned learned baselines.** `.\run_tuned.ps1` (about 40 minutes). It needs the base scores of steps 2 and 3 and writes
   `pilot16_*`, `pilot12tuned_*` and `pilot14tuned_*`.
5. **Further analyses.** `python pilot17_bp_fusion.py` (arterial-pressure fusions, needs step 4), `python trust_domain_cost.py`
   (device cost, needs `pilot13_redteam.csv` and `pilot15_redteam.csv`), `python diag_rr.py` (respiratory channels). `python bootstrap_cost_ci.py` (intervals of the red-team cost, needs `pilot13_cost.csv` and `pilot15_cost.csv`).
6. **Earlier configuration** (Bonferroni thresholds, untuned baselines; used for the arterial-pressure twin and the Bonferroni comparison):
   `python pilot9.py`, `python pilot7_baselines.py pilot9`, `python pilot8_combined.py pilot9`, `python pilot10_bp_twin.py`,
   and `python pilot11_redteam.py` (writes `pilot11_*`).
7. **Figures.** `python make_figures_v4.py` for the figures of the manuscript.

## Where the numbers of the paper come from

- Detection against the number of compromised sources (Figure 1): `results/pilot12tuned_comb_detection.csv`, `results/pilot14tuned_comb_detection.csv`.
- Red-team cost (Table 1 of the short version, Table 2 of the full version): `results/pilot13_cost.csv`, `results/pilot15_cost.csv`; the Bonferroni comparison: `results/pilot11_cost.csv`.
- Confidence intervals of the cost: `results/bootstrap_cost_ci.csv`.
- Cost in devices: `results/trust_domain_cost.csv`.
- Tuned baselines: `results/pilot16_chosen.csv`, `results/pilot16_validation.csv`.
- Arterial-pressure twin: `results/pilot10_bp_*.csv`; fusions: `results/pilot17_bp_fusion_detection.csv`.
- False-alarm audit of the redundancy detector: `results/pilot12_control_audit.csv`, `results/pilot14_control_audit.csv`.

## Notes and caveats

- The scripts were run by the authors in the order above on Windows. They have **not** been re-run from a clean checkout of this repository,
  so a path or environment difference is possible. The scripts that draw random numbers set a seed inside the script (`pilot6.py`, `pilot9.py`, `pilot7_baselines.py`,
  `pilot10_bp_twin.py`, `pilot11_redteam.py`, `pilot16_tuned_baselines.py`); `pilot8_combined.py` and `pilot17_bp_fusion.py` make no random draws (tied scores are
  handled as probabilities). Results can still vary slightly with library versions.
- The three `.ps1` drivers create `pilot12.py` and `pilot14.py` from `pilot9.py` by renaming the output prefix; these two files are generated, not stored here.
- `pilot6.py` is the common core, a pilot build on earlier pilots (2 to 5) whose scripts are not included; their results are not used in the paper.
- All attacks are synthetic. The recordings are from one centre.

## License

MIT (see `LICENSE`). VitalDB data are subject to their own terms of use.

## Citation

[Add the paper citation after acceptance.]
