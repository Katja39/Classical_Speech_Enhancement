import os
import re
import json
import numpy as np
import librosa
import soundfile as sf
from itertools import product

from parameter_ranges import param_ranges_ss, param_ranges_mmse, param_ranges_wiener, param_ranges_omlsa

from evaluation_metrics import calculate_pesq, calculate_stoi


def optimize_parameters(clean_reference, noisy_audio, sr, algorithm_function, param_ranges):
    print(f"\n{'=' * 60}")
    print("Parameter Optimization")
    print(f"{'=' * 60}")

    baseline_stoi = calculate_stoi(clean_reference, noisy_audio, sr) or 0
    baseline_pesq = calculate_pesq(clean_reference, noisy_audio, sr) or 0

    print(f"Baseline - STOI: {baseline_stoi:.4f}, PESQ: {baseline_pesq:.2f}")

    best_stoi = -1
    best_pesq = -1
    best_params_stoi = {}
    best_params_pesq = {}
    best_enhanced_stoi = None
    best_enhanced_pesq = None

    param_combinations = list(product(*param_ranges.values()))
    param_names = list(param_ranges.keys())
    total_combinations = len(param_combinations)

    print(f"Testing all {total_combinations} parameter combinations")
    print("-" * 50)

    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))

        if (i % max(1, total_combinations // 20) == 0) or (i < 5):
            sorted_items = sorted(param_dict.items())
            param_str = ", ".join([f"{k}={v}" for k, v in sorted_items])
            print(f"  Testing [{i + 1}/{total_combinations}]: {param_str}")

        try:
            # Apply algorithm
            enhanced = algorithm_function(noisy_audio, sr, **param_dict)

            if enhanced is None or len(enhanced) == 0:
                continue

            # Calculate stoi and pesq
            stoi_score = calculate_stoi(clean_reference, enhanced, sr)
            pesq_score = calculate_pesq(clean_reference, enhanced, sr)

            if stoi_score is None or pesq_score is None:
                continue

            epsilon_stoi = 1e-6
            epsilon_pesq = 1e-3

            stoi_improved = stoi_score > best_stoi + epsilon_stoi
            pesq_improved = pesq_score > best_pesq + epsilon_pesq

            if stoi_improved:
                best_stoi = stoi_score
                best_params_stoi = param_dict.copy()
                best_enhanced_stoi = enhanced.copy()

                sorted_best = sorted(param_dict.items())
                param_str = ", ".join([f"{k}={v}" for k, v in sorted_best])
                print(f"    New best STOI: {stoi_score:.4f}")
                print(f"    Parameters: {param_str}")

            if pesq_improved:
                best_pesq = pesq_score
                best_params_pesq = param_dict.copy()
                best_enhanced_pesq = enhanced.copy()

                sorted_best = sorted(param_dict.items())
                param_str = ", ".join([f"{k}={v}" for k, v in sorted_best])
                print(f"    New best PESQ: {pesq_score:.2f}")
                print(f"    Parameters: {param_str}")

        except Exception as e:
            print(f"  Warning with params {param_dict}: {e}")
            continue

        # Progress indicator
        if (i + 1) % max(1, total_combinations // 10) == 0:
            print(f"  Progress: {i + 1}/{total_combinations} | Best STOI: {best_stoi:.4f} | Best PESQ: {best_pesq:.2f}")

    print(f"\n{'=' * 60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'=' * 60}")

    print(f"\nOptimal Parameters for MAXIMIZING STOI:")
    print(f"   {'-' * 50}")

    if best_params_stoi:
        sorted_params = sorted(best_params_stoi.items())
        for param_name, param_value in sorted_params:

            if param_name in param_ranges:
                values = param_ranges[param_name]
                if len(values) > 1:
                    idx = values.index(param_value)
                    print(f"   {param_name:20} = {param_value} (Position {idx + 1}/{len(values)} in range {values})")
                else:
                    print(f"   {param_name:20} = {param_value}")
            else:
                print(f"   {param_name:20} = {param_value}")
    else:
        print("   No valid parameters found for STOI optimization")

    print(f"\n   STOI Score: {best_stoi:.4f}")
    print(
        f"   Improvement over baseline: {best_stoi - baseline_stoi:+.4f} ({100 * (best_stoi - baseline_stoi) / baseline_stoi:+.1f}%)")

    print(f"\nOptimal Parameters for MAXIMIZING PESQ (Audio-Qualität):")
    print(f"   {'-' * 50}")
    if best_params_pesq:
        sorted_params = sorted(best_params_pesq.items())
        for param_name, param_value in sorted_params:
            if param_name in param_ranges:
                values = param_ranges[param_name]
                if len(values) > 1:
                    idx = values.index(param_value)
                    print(f"   {param_name:20} = {param_value} (Position {idx + 1}/{len(values)} in range {values})")
                else:
                    print(f"   {param_name:20} = {param_value}")
            else:
                print(f"   {param_name:20} = {param_value}")
    else:
        print("   No valid parameters found for PESQ optimization")

    print(f"\n   PESQ Score: {best_pesq:.2f}")
    print(
        f"   Improvement over baseline: {best_pesq - baseline_pesq:+.2f} ({100 * (best_pesq - baseline_pesq) / baseline_pesq:+.1f}%)")

    print(f"\n{'=' * 60}")
    print("PARAMETER COMPARISON")
    print(f"{'=' * 60}")

    if best_params_stoi and best_params_pesq:
        if best_params_stoi == best_params_pesq:
            print("\n STOI and PESQ optimal parameters are IDENTICAL")
            print("   → Same parameters maximize both intelligibility and quality")
        else:
            print("\n STOI and PESQ optimal parameters are DIFFERENT")
            print("   → Trade-off between intelligibility and quality")

            print(f"\n   {'Parameter':20} {'STOI-optimal':15} {'PESQ-optimal':15} {'Difference'}")
            print(f"   {'-' * 20} {'-' * 15} {'-' * 15} {'-' * 10}")

            all_params = set(best_params_stoi.keys()) | set(best_params_pesq.keys())
            for param in sorted(all_params):
                stoi_val = best_params_stoi.get(param, "N/A")
                pesq_val = best_params_pesq.get(param, "N/A")
                if stoi_val != pesq_val:
                    diff_indicator = "↕️" if isinstance(stoi_val, (int, float)) and isinstance(pesq_val,
                                                                                               (int, float)) else "≠"
                    print(f"   {param:20} {str(stoi_val):15} {str(pesq_val):15} {diff_indicator}")
                else:
                    print(f"   {param:20} {str(stoi_val):15} {str(pesq_val):15} ✓")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total parameter combinations tested: {total_combinations}")
    print(
        f"Best STOI improvement: {best_stoi - baseline_stoi:+.4f} ({100 * (best_stoi - baseline_stoi) / baseline_stoi:+.1f}%)")
    print(
        f"Best PESQ improvement: {best_pesq - baseline_pesq:+.2f} ({100 * (best_pesq - baseline_pesq) / baseline_pesq:+.1f}%)")

    if best_params_stoi != best_params_pesq:
        num_different = sum(1 for k in set(best_params_stoi.keys()) | set(best_params_pesq.keys())
                            if best_params_stoi.get(k) != best_params_pesq.get(k))
        print(f"Parameters differing between STOI/PESQ optima: {num_different}/{len(param_names)}")

    if best_enhanced_stoi is None or best_enhanced_pesq is None:
        raise ValueError("Optimization failed - no valid parameters found!")

    print("=" * 60)

    return {
        'stoi': {'enhanced': best_enhanced_stoi, 'params': best_params_stoi, 'score': best_stoi},
        'pesq': {'enhanced': best_enhanced_pesq, 'params': best_params_pesq, 'score': best_pesq},
        'baseline': {'stoi': baseline_stoi, 'pesq': baseline_pesq},
        'improvements': {
            'stoi': best_stoi - baseline_stoi,
            'pesq': best_pesq - baseline_pesq,
            'stoi_percent': 100 * (best_stoi - baseline_stoi) / baseline_stoi if baseline_stoi > 0 else 0,
            'pesq_percent': 100 * (best_pesq - baseline_pesq) / baseline_pesq if baseline_pesq > 0 else 0
        }
    }

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


def run_algorithm_on_pair(alg_name, alg_fn, param_ranges, clean, noisy, sr, out_dir, stem):
    print(f"    Running optimization for {alg_name}...")

    def algorithm_wrapper(noisy_audio, sr, **params):
        # Check if noise_method is 'true_noise'
        if params.get('noise_method') == 'true_noise':
            try:
                return alg_fn(noisy_audio, sr, clean_audio=clean, **params)
            except TypeError as e:
                print(f"      Warning: Algorithm doesn't support true_noise, using estimation")
                return alg_fn(noisy_audio, sr, **params)
        else:
            return alg_fn(noisy_audio, sr, **params)

    opt = optimize_parameters(clean, noisy, sr, algorithm_wrapper, param_ranges)

    enhanced_stoi = opt["stoi"]["enhanced"]
    enhanced_pesq = opt["pesq"]["enhanced"]

    # Calculate metrics
    stoi_noisy = calculate_stoi(clean, noisy, sr)
    pesq_noisy = calculate_pesq(clean, noisy, sr)

    stoi_stoiopt = calculate_stoi(clean, enhanced_stoi, sr)
    pesq_stoiopt = calculate_pesq(clean, enhanced_stoi, sr)

    stoi_pesqopt = calculate_stoi(clean, enhanced_pesq, sr)
    pesq_pesqopt = calculate_pesq(clean, enhanced_pesq, sr)

    # Save files
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
        description='PRECISE batch comparison of speech enhancement algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--sample', type=int, default=0,
                        help='Number of files to sample (0 for all)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already processed files')
    parser.add_argument('--start-from', type=str, default='',
                        help='Start from specific file')
    parser.add_argument('--list-processed', action='store_true',
                        help='List already processed files and exit')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    out_ss = os.path.join(base_dir, "results_spectralSubtractor")
    out_mmse = os.path.join(base_dir, "results_mmse")
    out_wiener = os.path.join(base_dir, "results_wiener")
    out_omlsa = os.path.join(base_dir, "results_omlsa")
    summary_dir = os.path.join(base_dir, "results_summary")

    from spectral_subtractor import spectral_subtraction
    from mmse import mmse
    from wiener_filter import wiener_filter
    from advanced_mmse import advanced_mmse

    _ensure_dir(out_ss)
    _ensure_dir(out_mmse)
    _ensure_dir(out_wiener)
    _ensure_dir(out_omlsa)
    _ensure_dir(summary_dir)

    algorithms = [
        ("spectralSubtractor", spectral_subtraction, param_ranges_ss, out_ss),
        ("mmse", mmse, param_ranges_mmse, out_mmse),
        ("wiener", wiener_filter, param_ranges_wiener, out_wiener),
        ("omlsa", advanced_mmse, param_ranges_omlsa, out_omlsa),
    ]

    pairs = _find_pairs(data_dir)
    if not pairs:
        raise RuntimeError(
            "No file pairs found in ./data.\n"
        )

    # Sample files if requested
    if args.sample > 0 and args.sample < len(pairs):
        import random
        random.seed(42)
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

    # List processed files
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
        else:
            print("No processed data found.")

        original_count = len(pairs)

        # Filter already processed
        if args.resume and processed_stems:
            pairs = [p for p in pairs if p["stem"] not in processed_stems]
            print(f"Skipping {original_count - len(pairs)} already processed files")

        if args.start_from:
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
    print("PRECISE BATCH COMPARISON OF SPEECH ENHANCEMENT ALGORITHMS")
    print(f"Data folder : {data_dir}")
    print(f"Found pairs : {len(pairs)}")
    print("IMPORTANT: Using PRECISE optimization for each file (no shortcuts)")
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

    # Process all files
    print(f"\n{'=' * 60}")
    print(f"Processing all files with PRECISE optimization")
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

            # Run PRECISE optimization
            r = run_algorithm_on_pair(
                alg_name=alg_name,
                alg_fn=alg_fn,
                param_ranges=ranges,
                clean=clean,
                noisy=noisy,
                sr=target_sr,
                out_dir=out_dir,
                stem=stem
            )
            r["clean_path"] = p["clean"]
            r["noisy_path"] = p["noisy"]

            # Add result or update existing one
            if existing_entry_index >= 0:
                all_results[existing_entry_index] = r
                updated_results_count += 1
                print(f"    Updating existing entry for {stem}_{alg_name}")
            else:
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

    # Remove duplicates before saving
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
    print(f"\nSaving JSON to {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"JSON successfully saved ({len(all_results)} entries)")

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

    # CSV
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