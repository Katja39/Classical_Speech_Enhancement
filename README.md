# Classical Speech Enhancement
Study project for the course **Audio and Hearing Systems** (FH Erfurt). 
The repository implements classical speech‑enhancement algorithms, performs parameter optimisation on paired clean/noisy recordings, and evaluates objective speech quality and intelligibility (STOI, PESQ, SNR) including combined trade‑offs.

## Repository Overview
- `Code/` – Python implementation and analysis  
  - Algorithms: `spectral_subtractor.py`, `wiener_filter.py`, `mmse.py`, `advanced_mmse.py` (Log‑MMSE / OMLSA)  
  - Pipeline and grid search: `speech_enhancement_comparison.py`  
  - Metrics: `evaluation_metrics.py`  
  - Parameter grids: `parameter_ranges.py`  
  - Analysis and plotting: `evaluation/statistics.py` (JSON exports in `Code/evaluation/*.json`)
- `Document/` – scientific report (PDF and LaTeX sources)

## Requirements
- Python 3.13
- Packages: `numpy`, `scipy`, `librosa`, `soundfile`, `pandas`, `matplotlib`, `pesq`, `pystoi`

## Prepare Data
- Place paired files in `Code/data/`: `<name>_clean.wav` together with `<name>_noisy.wav` (or `<name>_noise.wav`).  
- Signals are resampled to 16 kHz and time‑aligned by cross‑correlation inside the pipeline.  

## Run Batch Experiments
Execute all commands inside `Code/` with the virtual environment active.

- Full experiment across all pairs:
```powershell
python speech_enhancement_comparison.py
```

- Optional arguments:  
  - `--list-processed` – list stems with existing outputs  
  - `--resume` – continue while skipping processed stems  
  - `--start-from <stem>` – begin at a specified stem (e.g., `p232_072`)

### Processing sequence
1) Load paired signals, convert to mono, resample to 16 kHz, and align noisy to clean.  
2) Run grid searches defined in `parameter_ranges.py` for each algorithm; generate three optimised variants:  
   - `*_optimized_stoi.wav` (STOI maximisation)  
   - `*_optimized_pesq.wav` (PESQ maximisation)  
   - `*_optimized_balanced.wav` (0.5·STOI + 0.5·PESQ_norm)  
3) Store per‑algorithm WAVs in `Code/results_<algorithm>/`.  
4) Write aggregates to `Code/results_summary/` (`all_results.json`, `all_results.csv`, `summary_means.json`).

## Evaluation and Plotting
- Script: `Code/evaluation/statistics.py` (plot blocks are toggled near the end of the file).  
- Run from `Code/`:
```powershell
python -m evaluation.statistics
```
- Figures render interactively; JSON summaries are emitted to `Code/evaluation/` according to the `output_json` parameters.

## Tuning Parameters
- Adjust `Code/parameter_ranges.py` to modify the search grid (FFT size, hop length, smoothing factors, noise estimation strategy).  
- `noise_method="true_noise"` employs the oracle noise track; alternative estimators (`percentile`, `min_tracking`) are prepared.

## Results
<details>
<summary>Click to expand</summary>
