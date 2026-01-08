import os
import re
import json
import random
import numpy as np
import librosa
import soundfile as sf
from itertools import product

from pystoi import stoi
import pesq

#Algorithms
from spectralSubtractor import spectral_subtraction
from mmse import mmse
from wiener_filter import wiener_filter
from advanced_mmse import advanced_mmse

#--resume
#--fast - faster, but less accurate

def calculate_pesq(clean_reference, test_audio, sr):
    try:
        min_length = min(len(clean_reference), len(test_audio))
        clean_trimmed = clean_reference[:min_length]
        test_trimmed = test_audio[:min_length]

        # PESQ requires 8000 Hz or 16000 Hz
        if sr == 16000:
            mode = 'wb'  # Wideband
            sr_pesq = 16000
        elif sr == 8000:
            mode = 'nb'  # Narrowband
            sr_pesq = 8000
        else:
            # Resample to 16000 Hz for PESQ if not supported
            clean_trimmed = librosa.resample(clean_trimmed, orig_sr=sr, target_sr=16000)
            test_trimmed = librosa.resample(test_trimmed, orig_sr=sr, target_sr=16000)
            mode = 'wb'
            sr_pesq = 16000

        return pesq.pesq(sr_pesq, clean_trimmed, test_trimmed, mode)
    except Exception as e:
        print(f"PESQ calculation failed: {e}")
        return None

def calculate_stoi(clean_reference, test_audio, sr):
    try:
        min_length = min(len(clean_reference), len(test_audio))
        return stoi(clean_reference[:min_length], test_audio[:min_length], sr, extended=False)
    except Exception as e:
        print(f"STOI calculation failed: {e}")
        return None

def optimize_parameters_smart(clean_reference, noisy_audio, sr, algorithm_function, param_ranges,
                              max_iterations=50, early_stop_threshold=0.7):
    """
    Intelligent multi-stage parameter optimization with early stopping
    Returns separate optimizations for STOI and PESQ
    """

    print(f"\n{'=' * 60}")
    print(f"Smart Parameter Optimization")
    print(f"{'=' * 60}")

    # Baseline STOI für noisy audio
    baseline_stoi = calculate_stoi(clean_reference, noisy_audio, sr) or 0
    baseline_pesq = calculate_pesq(clean_reference, noisy_audio, sr) or 0

    print(f"Baseline - STOI: {baseline_stoi:.4f}, PESQ: {baseline_pesq:.2f}")

    # STAGE 1: Coarse grid search for STOI
    print("\nStage 1: Coarse grid search (STOI)...")
    coarse_stoi_params, coarse_stoi_score, coarse_stoi_enhanced = _grid_search_with_early_stop(
        clean_reference, noisy_audio, sr, algorithm_function, param_ranges,
        baseline_score=baseline_stoi, threshold=early_stop_threshold,
        metric='stoi'
    )

    # STAGE 2: Local random search around best STOI
    print("\nStage 2: Local random search (STOI)...")
    best_stoi_params, best_stoi_score, best_stoi_enhanced = _random_search(
        clean_reference, noisy_audio, sr, algorithm_function,
        coarse_stoi_params, param_ranges, center_score=coarse_stoi_score,
        iterations=20, metric='stoi'
    )

    # STAGE 3: Coarse grid search for PESQ (starting from STOI-optimal)
    print("\nStage 3: Fine search around STOI-opt for PESQ...")
    best_pesq_params, best_pesq_score, best_pesq_enhanced = _random_search(
        clean_reference, noisy_audio, sr, algorithm_function,
        best_stoi_params, param_ranges, center_score=0,
        iterations=15, metric='pesq'
    )

    # Display results
    print(f"\nOptimal Parameters for STOI:")
    print(f"  {best_stoi_params}")
    print(f"  STOI: {best_stoi_score:.4f} (improvement: {best_stoi_score - baseline_stoi:+.4f})")

    print(f"\nOptimal Parameters for PESQ:")
    print(f"  {best_pesq_params}")
    print(f"  PESQ: {best_pesq_score:.2f} (improvement: {best_pesq_score - baseline_pesq:+.2f})")

    print("=" * 60)

    return {
        'stoi': {'enhanced': best_stoi_enhanced, 'params': best_stoi_params, 'score': best_stoi_score},
        'pesq': {'enhanced': best_pesq_enhanced, 'params': best_pesq_params, 'score': best_pesq_score}
    }


def _grid_search_with_early_stop(clean_reference, noisy_audio, sr, algorithm_function,
                                 param_ranges, baseline_score=0, threshold=0.7, metric='stoi'):
    """
    Grid search with early stopping based on threshold
    """
    best_score = -1
    best_params = {}
    best_enhanced = None

    # Create coarse parameter ranges
    coarse_ranges = {}
    for param, values in param_ranges.items():
        if len(values) > 4:
            # Take min, 25%, 50%, 75%, max
            indices = [0, len(values) // 4, len(values) // 2, 3 * len(values) // 4, -1]
            coarse_ranges[param] = [values[i] for i in indices if i < len(values)]
        elif len(values) > 2:
            # Take min, middle, max
            coarse_ranges[param] = [values[0], values[len(values) // 2], values[-1]]
        else:
            coarse_ranges[param] = values

    param_names = list(coarse_ranges.keys())
    param_combinations = list(product(*coarse_ranges.values()))
    total = len(param_combinations)

    print(f"  Testing {total} coarse combinations")

    evaluated = 0
    skipped = 0

    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))

        # Early skip for obviously bad parameters
        if _should_skip_parameters(param_dict, algorithm_function):
            skipped += 1
            continue

        enhanced = algorithm_function(noisy_audio, sr, **param_dict)

        if metric == 'stoi':
            score = calculate_stoi(clean_reference, enhanced, sr) or 0
        else:
            score = calculate_pesq(clean_reference, enhanced, sr) or 0

        evaluated += 1

        # Early stopping: if score is too low compared to baseline
        if score < baseline_score * threshold and metric == 'stoi':
            continue

        if score > best_score:
            best_score = score
            best_params = param_dict
            best_enhanced = enhanced

        # Progress indicator
        if (i + 1) % max(1, total // 10) == 0:
            print(f"    Progress: {i + 1}/{total}, best {metric.upper()}: {best_score:.4f}")

    print(f"  Evaluated: {evaluated}, Skipped: {skipped}, Best: {best_score:.4f}")
    return best_params, best_score, best_enhanced


def _random_search(clean_reference, noisy_audio, sr, algorithm_function,
                   center_params, full_ranges, center_score=0,
                   iterations=30, metric='stoi'):
    """
    Random search around center parameters
    """
    best_params = center_params.copy()
    best_enhanced = algorithm_function(noisy_audio, sr, **best_params)

    if metric == 'stoi':
        best_score = calculate_stoi(clean_reference, best_enhanced, sr) or 0
    else:
        best_score = calculate_pesq(clean_reference, best_enhanced, sr) or 0

    print(f"  Starting from {metric.upper()}: {best_score:.4f}")

    improvement_count = 0

    for i in range(iterations):
        # Generate trial parameters
        trial_params = {}
        for param, values in full_ranges.items():
            if param in center_params and len(values) > 1:
                current_val = center_params[param]

                # Try to find current value in range
                if current_val in values:
                    idx = values.index(current_val)
                else:
                    # Find closest value
                    idx = min(range(len(values)), key=lambda i: abs(values[i] - current_val))

                # Move randomly within +/- 2 positions, but stay in bounds
                max_step = max(1, len(values) // 4)  # Don't move too far
                step = random.randint(-max_step, max_step)
                new_idx = max(0, min(len(values) - 1, idx + step))
                trial_params[param] = values[new_idx]
            else:
                # Random choice for parameters not in center
                trial_params[param] = random.choice(values)

        # Skip if parameters are identical to current best
        if trial_params == best_params:
            continue

        enhanced = algorithm_function(noisy_audio, sr, **trial_params)

        if metric == 'stoi':
            score = calculate_stoi(clean_reference, enhanced, sr) or 0
        else:
            score = calculate_pesq(clean_reference, enhanced, sr) or 0

        # Accept if better
        if score > best_score:
            best_score = score
            best_params = trial_params
            best_enhanced = enhanced
            improvement_count += 1

            # If we found significant improvement, we can be more aggressive
            if improvement_count >= 3:
                # Reset counter and continue
                improvement_count = 0

    print(f"  Finished with {metric.upper()}: {best_score:.4f}")
    return best_params, best_score, best_enhanced


def _should_skip_parameters(param_dict, algorithm_function):
    """
    Heuristic to skip obviously bad parameter combinations
    """
    func_name = algorithm_function.__name__.lower()

    # Skip overly aggressive spectral subtraction
    if 'spectral' in func_name:
        alpha = param_dict.get('alpha', 1.0)
        beta = param_dict.get('beta', 0.01)

        # Very high alpha with very low beta causes too much distortion
        if alpha > 4.0 and beta < 0.005:
            return True

        # Very low alpha won't remove enough noise
        if alpha < 0.5:
            return True

    # Skip extreme values for MMSE algorithms
    if 'mmse' in func_name or 'wiener' in func_name:
        alpha = param_dict.get('alpha', 0.98)

        # Alpha too low leads to slow adaptation
        if alpha < 0.8:
            return True

        # Alpha too high leads to musical noise
        if alpha > 0.995:
            return True

        # Very high gain floor won't reduce noise enough
        gain_floor = param_dict.get('gain_floor', 0.001)
        if gain_floor > 0.1:
            return True

    return False


def optimize_parameters_fast(clean_reference, noisy_audio, sr, algorithm_function, param_ranges):
    """
    Fast version for batch processing - uses smart optimization with reduced iterations
    """
    # Use smart optimization but with fewer iterations
    return optimize_parameters_smart(
        clean_reference, noisy_audio, sr, algorithm_function, param_ranges,
        max_iterations=30,  # Reduced from 50
        early_stop_threshold=0.6  # Slightly lower threshold
    )


def _find_pairs(data_dir: str):
    wavs = [f for f in os.listdir(data_dir) if f.lower().endswith(".wav")]
    clean_files = [f for f in wavs if "_clean" in f.lower()]

    pairs = []
    for cf in clean_files:
        stem = re.sub(r"(?i)_clean\.wav$", "", cf)

        candidates = [
            f"{stem}_noisy.wav",
            f"{stem}_noise.wav",
            f"{stem}_noiseWithMusic.wav",
            f"{stem}_noisewithmusic.wav",
        ]

        fallback = [f for f in wavs
                    if f.lower().startswith(stem.lower())
                    and ("noise" in f.lower() or "noisy" in f.lower())
                    and f.lower() != cf.lower()]

        noisy = None
        for c in candidates:
            if c in wavs:
                noisy = c
                break
        if noisy is None and len(fallback) == 1:
            noisy = fallback[0]

        if noisy is not None:
            pairs.append({
                "stem": stem,
                "clean": os.path.join(data_dir, cf),
                "noisy": os.path.join(data_dir, noisy)
            })

    return pairs


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _resample_if_needed(x, sr_x, sr_target):
    if sr_x == sr_target:
        return x
    return librosa.resample(x, orig_sr=sr_x, target_sr=sr_target)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _fmt(x, digits=4):
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def run_algorithm_on_pair(alg_name, alg_fn, param_ranges, clean, noisy, sr, out_dir, stem, use_fast=True):
    """
    Run algorithm with smart optimization
    """
    if use_fast:
        print(f"    Using FAST optimization...")
        opt = optimize_parameters_fast(clean, noisy, sr, alg_fn, param_ranges)
    else:
        print(f"    Using SMART optimization...")
        opt = optimize_parameters_smart(clean, noisy, sr, alg_fn, param_ranges)

    enhanced_stoi = opt["stoi"]["enhanced"]
    enhanced_pesq = opt["pesq"]["enhanced"]

    stoi_noisy = calculate_stoi(clean, noisy, sr)
    pesq_noisy = calculate_pesq(clean, noisy, sr)

    stoi_stoiopt = calculate_stoi(clean, enhanced_stoi, sr)
    pesq_stoiopt = calculate_pesq(clean, enhanced_stoi, sr)

    stoi_pesqopt = calculate_stoi(clean, enhanced_pesq, sr)
    pesq_pesqopt = calculate_pesq(clean, enhanced_pesq, sr)

    _ensure_dir(out_dir)
    path_stoi = os.path.join(out_dir, f"{stem}_{alg_name}_optimized_stoi.wav")
    path_pesq = os.path.join(out_dir, f"{stem}_{alg_name}_optimized_pesq.wav")
    sf.write(path_stoi, enhanced_stoi, sr)
    sf.write(path_pesq, enhanced_pesq, sr)

    return {
        "alg": alg_name,
        "stem": stem,
        "sr": sr,

        "stoi_noisy": stoi_noisy,
        "pesq_noisy": pesq_noisy,
        "stoi_stoiopt": stoi_stoiopt,
        "pesq_stoiopt": pesq_stoiopt,
        "stoi_pesqopt": stoi_pesqopt,
        "pesq_pesqopt": pesq_pesqopt,

        "best_params_stoi": opt["stoi"].get("params", {}),
        "best_params_pesq": opt["pesq"].get("params", {}),

        "enhanced_path_stoi": path_stoi,
        "enhanced_path_pesq": path_pesq,

        "clean_path": None,
        "noisy_path": None,
    }

def _compute_and_save_summary(all_results, algorithms, summary_dir):
    """
    Compute and save summary from existing results
    """
    print("\nComputing summary from existing results...")

    summary = {}
    for alg_name, _, _, _ in algorithms:
        rows = [r for r in all_results if r["alg"] == alg_name]
        summary[alg_name] = {
            "count": len(rows),
            "stoi_noisy_mean": _mean([r["stoi_noisy"] for r in rows]),
            "pesq_noisy_mean": _mean([r["pesq_noisy"] for r in rows]),
            "stoi_stoiopt_mean": _mean([r["stoi_stoiopt"] for r in rows]),
            "pesq_stoiopt_mean": _mean([r["pesq_stoiopt"] for r in rows]),
            "stoi_pesqopt_mean": _mean([r["stoi_pesqopt"] for r in rows]),
            "pesq_pesqopt_mean": _mean([r["pesq_pesqopt"] for r in rows]),
        }

    summary_path = os.path.join(summary_dir, "summary_means.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to {summary_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY FROM EXISTING RESULTS")
    print("=" * 70)
    for alg_name in summary:
        s = summary[alg_name]
        print(f"\n{alg_name.upper()}  (N={s['count']})")
        print(f"  STOI noisy mean      : {_fmt(s['stoi_noisy_mean'], 4)}")
        print(f"  STOI STOI-opt mean   : {_fmt(s['stoi_stoiopt_mean'], 4)}")
        print(f"  STOI PESQ-opt mean   : {_fmt(s['stoi_pesqopt_mean'], 4)}")
        print(f"  PESQ noisy mean      : {_fmt(s['pesq_noisy_mean'], 2)}")
        print(f"  PESQ STOI-opt mean   : {_fmt(s['pesq_stoiopt_mean'], 2)}")
        print(f"  PESQ PESQ-opt mean   : {_fmt(s['pesq_pesqopt_mean'], 2)}")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch comparison of speech enhancement algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Performance-Optimization
    parser.add_argument('--fast', action='store_true',
                        help='Use fast optimization mode (15x faster, slightly less accurate)')

    # Sampling/Testing
    parser.add_argument('--sample', type=int, default=0,
                        help='Number of files to sample (0 for all, e.g., --sample 50 for testing)')

    # Parameter-Optimization
    parser.add_argument('--optimize-on-sample', type=int, default=5,
                      help='Find optimal parameters on N files, then apply to all (e.g., --optimize-on-sample 10)')

    # Resume
    parser.add_argument('--resume', action='store_true',
                        help='Skip already processed files (detects existing .wav results)')

    parser.add_argument('--start-from', type=str, default='',
                        help='Start from specific file (e.g., --start-from p232_027)')

    parser.add_argument('--list-processed', action='store_true',
                        help='List already processed files and exit')

    # Debug/Info
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed progress information')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    out_ss = os.path.join(base_dir, "results_spectralSubtractor")
    out_mmse = os.path.join(base_dir, "results_mmse")
    out_wiener = os.path.join(base_dir, "results_wiener")
    out_omlsa = os.path.join(base_dir, "results_omlsa")
    summary_dir = os.path.join(base_dir, "results_summary")

    _ensure_dir(out_ss)
    _ensure_dir(out_mmse)
    _ensure_dir(out_wiener)
    _ensure_dir(out_omlsa)
    _ensure_dir(summary_dir)

    ranges_ss = {
        "alpha": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "beta": [0.001, 0.005, 0.01, 0.02, 0.05],
        "n_fft": [512, 1024],
        "hop_length": [128, 256],
    }

    ranges_mmse = {
        "alpha": [0.85, 0.92, 0.96, 0.98],
        "n_fft": [512, 1024],
        "hop_length": [128, 256],
        "ksi_min": [0.001, 0.01, 0.05],
    }

    ranges_wiener = {
        "alpha": [0.85, 0.92, 0.96, 0.98],
        "n_fft": [512, 1024],
        "hop_length": [128, 256],
        "gain_floor": [0.0005, 0.001, 0.005, 0.01],
    }

    ranges_omlsa = {
        "alpha": [0.92, 0.96, 0.98],
        "n_fft": [512, 1024],
        "hop_length": [128, 256],
        "ksi_min": [0.001, 0.01, 0.05],
        "gain_floor": [0.02, 0.05, 0.1],
        "noise_mu": [0.95, 0.98],
        "q": [0.3, 0.5, 0.7],
    }

    algorithms = [
        ("spectralSubtractor", spectral_subtraction, ranges_ss, out_ss),
        ("mmse", mmse, ranges_mmse, out_mmse),
        ("wiener", wiener_filter, ranges_wiener, out_wiener),
        ("omlsa", advanced_mmse, ranges_omlsa, out_omlsa),
    ]

    pairs = _find_pairs(data_dir)
    if not pairs:
        raise RuntimeError(
            "No file pairs found in ./data.\n"
            "Expected e.g. <stem>_clean.wav and <stem>_noiseWithMusic.wav (or _noisy.wav)."
        )

    # Sample files if requested
    if args.sample > 0 and args.sample < len(pairs):
        import random
        random.seed(42)  # For reproducibility
        pairs = random.sample(pairs, args.sample)
        print(f"Sampling {args.sample} files from dataset")

    target_sr = 16000

    # Find already processed files
    def get_processed_stems():
        processed = set()
        result_dirs = [out_ss, out_mmse, out_wiener, out_omlsa]

        for result_dir in result_dirs:
            if os.path.exists(result_dir):
                for file in os.listdir(result_dir):
                    if file.endswith('.wav') and ('_stoi.wav' in file or '_pesq.wav' in file):
                        parts = file.split('_')
                        if len(parts) >= 2:
                            stem = '_'.join(parts[:2])
                            processed.add(stem)
        return processed

    # List of processed data
    if args.list_processed:
        processed = get_processed_stems()
        print(f"Already processed data ({len(processed)}):")
        for stem in sorted(processed):
            print(f"  {stem}")
        return

    # Resume logic
    if args.resume or args.start_from:
        print("\n" + "=" * 60)
        print("Resume mode")
        print("=" * 60)

        processed_stems = get_processed_stems()

        if processed_stems:
            print(f"Found: {len(processed_stems)} already processed data")
            print(f"Examples: {sorted(list(processed_stems))[:5]}")
        else:
            print("No processed data found.")

        original_count = len(pairs)

        # Filter already processed
        if args.resume and processed_stems:
            pairs = [p for p in pairs if p["stem"] not in processed_stems]
            print(
                f"Skipping {original_count - len(pairs)} already processed files (based on audio files)")

        if args.start_from:
            # index start data
            start_index = 0
            for i, p in enumerate(pairs):
                if p["stem"] == args.start_from:
                    start_index = i
                    break

            if start_index > 0:
                pairs = pairs[start_index:]
                print(f"Begin: {args.start_from} (Index {start_index})")
                print(f"Skip {start_index} previous files")
            else:
                print(f"Start file '{args.start_from}' not found. Starting from the beginning.")

        print(f"Remaining files to process: {len(pairs)}/{original_count}")

        if len(pairs) == 0:
            print("No new files to process.")

            json_path = os.path.join(summary_dir, "all_results.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        all_results = json.load(f)
                    print(f"\nLoading existing results ({len(all_results)} entries) for summary...")
                    # Calculate and show summary
                    _compute_and_save_summary(all_results, algorithms, summary_dir)
                except Exception as e:
                    print(f"Error loading existing results: {e}")
            return

        confirm = input("\nContinue? (y/n): ")
        if confirm.lower() != 'y':
            print("Aborted")
            return

        print("=" * 60 + "\n")

    all_results = []

    print("=" * 70)
    print("BATCH COMPARISON OF SPEECH ENHANCEMENT ALGORITHMS")
    print(f"Data folder : {data_dir}")
    print(f"Found pairs : {len(pairs)}")
    print(f"Optimization mode: {'FAST' if args.fast else 'SMART'}")
    if args.sample > 0:
        print(f"Sampling: {args.sample} files")
    print("=" * 70)

    # Load existing results from JSON
    json_path = os.path.join(summary_dir, "all_results.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            print(f"\nExisting JSON file found: {len(existing_results)} entries")

            if args.resume:
                # Keep existing results
                all_results = existing_results
                print(f"Keeping {len(all_results)} existing results (--resume active)")

                # Filter out stems already contained in JSON from pairs
                existing_stems_in_json = {r["stem"] for r in existing_results if "stem" in r}
                if existing_stems_in_json:
                    original_count = len(pairs)
                    pairs = [p for p in pairs if p["stem"] not in existing_stems_in_json]
                    print(f"Additionally skipping {original_count - len(pairs)} files already contained in JSON")
            else:
                print("Starting with empty results (--resume not active)")
                all_results = []

        except Exception as e:
            print(f"Error loading existing JSON: {e}")
            print("Starting with empty results")
            all_results = []
    else:
        print("\nNo existing JSON file found. Starting with empty results.")
        all_results = []

    # Phase 1: Find optimal parameters on a small sample
    if args.optimize_on_sample > 0 and args.optimize_on_sample < len(pairs):
        print(f"\n{'=' * 60}")
        print(f"PHASE 1: Finding optimal parameters on {args.optimize_on_sample} sample files")
        print(f"{'=' * 60}")

        sample_pairs = pairs[:args.optimize_on_sample]
        optimal_params = {}

        for alg_name, alg_fn, ranges, _ in algorithms:
            print(f"\nFinding optimal parameters for {alg_name}...")

            # Aggregate results from sample files
            all_scores_stoi = []
            all_params_stoi = []
            all_scores_pesq = []
            all_params_pesq = []

            for i, p in enumerate(sample_pairs, 1):
                print(f"  Sample {i}/{len(sample_pairs)}: {p['stem']}")

                clean, sr_c = librosa.load(p["clean"], sr=None)
                noisy, sr_n = librosa.load(p["noisy"], sr=None)

                clean = _resample_if_needed(clean, sr_c, target_sr)
                noisy = _resample_if_needed(noisy, sr_n, target_sr)

                L = min(len(clean), len(noisy))
                clean = clean[:L]
                noisy = noisy[:L]

                # Use fast optimization for parameter finding
                opt = optimize_parameters_fast(clean, noisy, target_sr, alg_fn, ranges)

                all_scores_stoi.append(opt['stoi']['score'])
                all_params_stoi.append(opt['stoi']['params'])

                all_scores_pesq.append(opt['pesq']['score'])
                all_params_pesq.append(opt['pesq']['params'])

            # Find most common/best parameters
            # Simple approach: take parameters from best performing sample
            if all_scores_stoi:
                best_idx_stoi = np.argmax(all_scores_stoi)
                best_idx_pesq = np.argmax(all_scores_pesq)

                optimal_params[alg_name] = {
                    'stoi': all_params_stoi[best_idx_stoi],
                    'pesq': all_params_pesq[best_idx_pesq]
                }

                print(f"  Selected STOI params: {optimal_params[alg_name]['stoi']}")
                print(f"  Selected PESQ params: {optimal_params[alg_name]['pesq']}")

    # Phase 2: Process all files
    print(f"\n{'=' * 60}")
    print(f"PHASE 2: Processing all files")
    print(f"{'=' * 60}")

    new_results_count = 0
    updated_results_count = 0

    for i, p in enumerate(pairs, 1):
        stem = p["stem"]
        print("\n" + "-" * 70)
        print(f"[{i}/{len(pairs)}] Processing: {stem}")
        print(f"  clean: {os.path.basename(p['clean'])}")
        print(f"  noisy: {os.path.basename(p['noisy'])}")
        print("-" * 70)

        clean, sr_c = librosa.load(p["clean"], sr=None)
        noisy, sr_n = librosa.load(p["noisy"], sr=None)

        clean = _resample_if_needed(clean, sr_c, target_sr)
        noisy = _resample_if_needed(noisy, sr_n, target_sr)

        L = min(len(clean), len(noisy))
        clean = clean[:L]
        noisy = noisy[:L]

        for alg_name, alg_fn, ranges, out_dir in algorithms:
            print(f"  -> {alg_name}:")

            # Check whether this entry already exists in all_results
            existing_entry_index = -1
            for idx, result in enumerate(all_results):
                if result.get("stem") == stem and result.get("alg") == alg_name:
                    existing_entry_index = idx
                    break

            # If we found optimal parameters in phase 1, use them directly
            if 'optimal_params' in locals() and alg_name in optimal_params:
                print(f"    Using pre-optimized parameters...")

                # Apply STOI-optimized parameters
                enhanced_stoi = alg_fn(noisy, target_sr, **optimal_params[alg_name]['stoi'])
                stoi_score_stoi = calculate_stoi(clean, enhanced_stoi, target_sr)

                # Apply PESQ-optimized parameters
                enhanced_pesq = alg_fn(noisy, target_sr, **optimal_params[alg_name]['pesq'])
                pesq_score_pesq = calculate_pesq(clean, enhanced_pesq, target_sr)

                stoi_noisy = calculate_stoi(clean, noisy, target_sr)
                pesq_noisy = calculate_pesq(clean, noisy, target_sr)
                stoi_score_pesq = calculate_stoi(clean, enhanced_pesq, target_sr)
                pesq_score_stoi = calculate_pesq(clean, enhanced_stoi, target_sr)

                _ensure_dir(out_dir)
                path_stoi = os.path.join(out_dir, f"{stem}_{alg_name}_optimized_stoi.wav")
                path_pesq = os.path.join(out_dir, f"{stem}_{alg_name}_optimized_pesq.wav")
                sf.write(path_stoi, enhanced_stoi, target_sr)
                sf.write(path_pesq, enhanced_pesq, target_sr)

                r = {
                    "alg": alg_name,
                    "stem": stem,
                    "sr": target_sr,
                    "stoi_noisy": stoi_noisy,
                    "pesq_noisy": pesq_noisy,
                    "stoi_stoiopt": stoi_score_stoi,
                    "pesq_stoiopt": pesq_score_stoi,
                    "stoi_pesqopt": stoi_score_pesq,
                    "pesq_pesqopt": pesq_score_pesq,
                    "best_params_stoi": optimal_params[alg_name]['stoi'],
                    "best_params_pesq": optimal_params[alg_name]['pesq'],
                    "enhanced_path_stoi": path_stoi,
                    "enhanced_path_pesq": path_pesq,
                    "clean_path": p["clean"],
                    "noisy_path": p["noisy"],
                }
            else:
                # Otherwise do full optimization
                r = run_algorithm_on_pair(
                    alg_name=alg_name,
                    alg_fn=alg_fn,
                    param_ranges=ranges,
                    clean=clean,
                    noisy=noisy,
                    sr=target_sr,
                    out_dir=out_dir,
                    stem=stem,
                    use_fast=args.fast
                )
                r["clean_path"] = p["clean"]
                r["noisy_path"] = p["noisy"]

            # Add result or update existing one
            if existing_entry_index >= 0:
                # Update existing entry
                all_results[existing_entry_index] = r
                updated_results_count += 1
                print(f"    Updating existing entry for {stem}_{alg_name}")
            else:
                # Add new entry
                all_results.append(r)
                new_results_count += 1
                print(f"    Adding new entry for {stem}_{alg_name}")

            print(
                f"     STOI noisy {_fmt(r['stoi_noisy'], 4)} | "
                f"STOI(STOI-opt) {_fmt(r['stoi_stoiopt'], 4)} | "
                f"STOI(PESQ-opt) {_fmt(r['stoi_pesqopt'], 4)}"
            )
            print(
                f"     PESQ noisy {_fmt(r['pesq_noisy'], 2)} | "
                f"PESQ(STOI-opt) {_fmt(r['pesq_stoiopt'], 2)} | "
                f"PESQ(PESQ-opt) {_fmt(r['pesq_pesqopt'], 2)}"
            )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"New entries: {new_results_count}")
    print(f"Updated entries: {updated_results_count}")
    print(f"Total entries in JSON: {len(all_results)}")

    # Remove duplicates before saving (safety measure)
    unique_results = {}
    for result in all_results:
        key = f"{result['stem']}_{result['alg']}"
        unique_results[key] = result

    all_results = list(unique_results.values())
    print(f"Unique entries after duplicate cleanup: {len(all_results)}")

    # Sort for better readability
    all_results.sort(key=lambda x: (x['stem'], x['alg']))

    # Save JSON
    json_path = os.path.join(summary_dir, "all_results.json")
    print(f"\nSpeichere JSON nach {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"JSON erfolgreich gespeichert ({len(all_results)} Einträge)")

    # Compute summary means
    summary = {}
    for alg_name, _, _, _ in algorithms:
        rows = [r for r in all_results if r["alg"] == alg_name]
        summary[alg_name] = {
            "count": len(rows),
            "stoi_noisy_mean": _mean([r["stoi_noisy"] for r in rows]),
            "pesq_noisy_mean": _mean([r["pesq_noisy"] for r in rows]),
            "stoi_stoiopt_mean": _mean([r["stoi_stoiopt"] for r in rows]),
            "pesq_stoiopt_mean": _mean([r["pesq_stoiopt"] for r in rows]),
            "stoi_pesqopt_mean": _mean([r["stoi_pesqopt"] for r in rows]),
            "pesq_pesqopt_mean": _mean([r["pesq_pesqopt"] for r in rows]),
        }

    summary_path = os.path.join(summary_dir, "summary_means.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV (None-safe)
    csv_path = os.path.join(summary_dir, "all_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = [
            "stem", "alg", "stoi_noisy", "pesq_noisy",
            "stoi_stoiopt", "pesq_stoiopt", "stoi_pesqopt", "pesq_pesqopt",
            "enhanced_path_stoi", "enhanced_path_pesq", "clean_path", "noisy_path"
        ]
        f.write(",".join(header) + "\n")

        def cell(x, digits=6):
            if x is None:
                return ""
            return f"{x:.{digits}f}"

        for r in all_results:
            row = [
                r["stem"], r["alg"],
                cell(r["stoi_noisy"]), cell(r["pesq_noisy"]),
                cell(r["stoi_stoiopt"]), cell(r["pesq_stoiopt"]),
                cell(r["stoi_pesqopt"]), cell(r["pesq_pesqopt"]),
                r["enhanced_path_stoi"], r["enhanced_path_pesq"],
                r["clean_path"], r["noisy_path"],
            ]
            f.write(",".join(row) + "\n")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL AVERAGE RESULTS (MEAN OVER ALL FILES)")
    print("=" * 70)
    for alg_name in summary:
        s = summary[alg_name]
        print(f"\n{alg_name.upper()}  (N={s['count']})")
        print(f"  STOI noisy mean      : {_fmt(s['stoi_noisy_mean'], 4)}")
        print(f"  STOI STOI-opt mean   : {_fmt(s['stoi_stoiopt_mean'], 4)}")
        print(f"  STOI PESQ-opt mean   : {_fmt(s['stoi_pesqopt_mean'], 4)}")
        print(f"  PESQ noisy mean      : {_fmt(s['pesq_noisy_mean'], 2)}")
        print(f"  PESQ STOI-opt mean   : {_fmt(s['pesq_stoiopt_mean'], 2)}")
        print(f"  PESQ PESQ-opt mean   : {_fmt(s['pesq_pesqopt_mean'], 2)}")

    print("\nSaved:")
    print(f"  Per-file results JSON : {json_path}")
    print(f"  Mean summary JSON     : {summary_path}")
    print(f"  Per-file results CSV  : {csv_path}")
    print("\nEnhanced audio folders:")
    print(f"  {out_ss}")
    print(f"  {out_mmse}")
    print(f"  {out_wiener}")
    print(f"  {out_omlsa}")


if __name__ == "__main__":
    main()