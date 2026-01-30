import os
import re
import json
import numpy as np
import librosa
import soundfile as sf
from itertools import product
from typing import Dict, Any

from parameter_ranges import param_ranges_ss, param_ranges_mmse, param_ranges_wiener, param_ranges_omlsa
from evaluation_metrics import calculate_pesq, calculate_stoi, calculate_snr, calculate_combined_speech_score

def optimize_parameters(clean_reference, noisy_audio, sr, algorithm_function, param_ranges) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print("Parameter Optimization")
    print(f"{'=' * 60}")

    baseline_stoi = calculate_stoi(clean_reference, noisy_audio, sr) or 0
    baseline_pesq = calculate_pesq(clean_reference, noisy_audio, sr) or 0
    baseline_snr = calculate_snr(clean_reference, noisy_audio) or 0
    baseline_comp = calculate_combined_speech_score(baseline_stoi, baseline_pesq)

    print(f"Baseline - STOI: {baseline_stoi:.4f}, PESQ: {baseline_pesq:.2f}, "
          f"Balance: {baseline_comp:.4f}, SNR: {baseline_snr:.2f} dB")

    results = {
        'stoi': {
            'score': -1,
            'params': {},
            'enhanced': None,
            'pesq': 0,
            'snr': 0
        },
        'pesq': {
            'score': -1,
            'params': {},
            'enhanced': None,
            'stoi': 0,
            'snr': 0
        },
        'balance': {
            'score': -1,
            'params': {},
            'enhanced': None,
            'stoi': 0,
            'pesq': 0,
            'snr': 0
        }
    }

    param_combinations = list(product(*param_ranges.values()))
    param_names = list(param_ranges.keys())
    total_combinations = len(param_combinations)

    print(f"Testing {total_combinations} parameter combinations")
    print("-" * 50)

    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))

        if (i % max(1, total_combinations // 20) == 0) or (i < 5):
            sorted_items = sorted(param_dict.items())
            param_str = ", ".join([f"{k}={v}" for k, v in sorted_items])
            print(f" Testing [{i + 1}/{total_combinations}]: {param_str}")

        try:
            enhanced = algorithm_function(noisy_audio, sr, **param_dict)
            if enhanced is None or len(enhanced) == 0:
                continue

            stoi_score = calculate_stoi(clean_reference, enhanced, sr)
            pesq_score = calculate_pesq(clean_reference, enhanced, sr)

            if stoi_score is None or pesq_score is None:
                continue

            comp_score = calculate_combined_speech_score(stoi_score, pesq_score)
            current_snr = calculate_snr(clean_reference, enhanced)

            if stoi_score > results['stoi']['score'] + 1e-6:
                results['stoi'].update({
                    'score': stoi_score,
                    'params': param_dict.copy(),
                    'enhanced': enhanced.copy(),
                    'pesq': pesq_score,
                    'snr': current_snr
                })
                print(f" New best STOI: {stoi_score:.4f} (PESQ: {pesq_score:.2f})")

            if pesq_score > results['pesq']['score'] + 1e-3:
                results['pesq'].update({
                    'score': pesq_score,
                    'params': param_dict.copy(),
                    'enhanced': enhanced.copy(),
                    'stoi': stoi_score,
                    'snr': current_snr
                })
                print(f" New best PESQ: {pesq_score:.2f} (STOI: {stoi_score:.4f})")

            if comp_score > results['balance']['score'] + 1e-5:
                results['balance'].update({
                    'score': comp_score,
                    'params': param_dict.copy(),
                    'enhanced': enhanced.copy(),
                    'stoi': stoi_score,
                    'pesq': pesq_score,
                    'snr': current_snr
                })
                print(f" New best BALANCE: {comp_score:.4f} "
                      f"(STOI: {stoi_score:.4f}, PESQ: {pesq_score:.2f})")

        except Exception as e:
            print(f" Warning with params {param_dict}: {e}")
            continue

        if (i + 1) % max(1, total_combinations // 10) == 0:
            print(f" Progress: {i + 1}/{total_combinations} | "
                  f"Best STOI: {results['stoi']['score']:.4f} | "
                  f"Best PESQ: {results['pesq']['score']:.2f} | "
                  f"Best Bal: {results['balance']['score']:.4f}")

    print(f"\n{'=' * 60}\nOPTIMIZATION RESULTS\n{'=' * 60}")
    print(f"Best STOI: {results['stoi']['score']:.4f} | "
          f"Best PESQ: {results['pesq']['score']:.2f} | "
          f"Best Balance: {results['balance']['score']:.4f}")

    for key in ['stoi', 'pesq', 'balance']:
        if results[key]['enhanced'] is None:
            raise ValueError(f"Optimization failed for {key} - no valid parameters found!")

    return {
        'stoi': results['stoi'],
        'pesq': results['pesq'],
        'balance': results['balance'],
        'baseline': {
            'stoi': baseline_stoi,
            'pesq': baseline_pesq,
            'snr': baseline_snr,
            'balance': baseline_comp
        },
        'improvements': {
            'stoi': results['stoi']['score'] - baseline_stoi,
            'pesq': results['pesq']['score'] - baseline_pesq,
            'balance': results['balance']['score'] - baseline_comp
        }
    }

def _find_pairs(data_dir: str):
    wavs = [f for f in os.listdir(data_dir) if f.lower().endswith(".wav")]
    clean_files = [f for f in wavs if "_clean" in f.lower()]
    pairs = []
    for cf in clean_files:
        stem = re.sub(r"(?i)_clean\.wav$", "", cf)
        candidates = [f"{stem}_noisy.wav", f"{stem}_noise.wav", f"{stem}_noiseWithMusic.wav",
                      f"{stem}_noisewithmusic.wav"]
        fallback = [f for f in wavs if f.lower().startswith(stem.lower()) and (
                    "noise" in f.lower() or "noisy" in f.lower()) and f.lower() != cf.lower()]
        noisy = next((c for c in candidates if c in wavs), fallback[0] if len(fallback) == 1 else None)
        if noisy:
            pairs.append({"stem": stem, "clean": os.path.join(data_dir, cf), "noisy": os.path.join(data_dir, noisy)})
    return pairs

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _resample_if_needed(x, sr_x, sr_target):
    if sr_x == sr_target: return x
    return librosa.resample(x, orig_sr=sr_x, target_sr=sr_target)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _fmt(x, digits=4):
    if x is None: return "NA"
    return f"{x:.{digits}f}"

def run_algorithm_on_pair(alg_name, alg_fn, param_ranges, clean, noisy, sr, out_dir, stem):
    print(f" Running optimization for {alg_name}...")

    def algorithm_wrapper(noisy_audio, sr, **params):
        if params.get('noise_method') == 'true_noise':
            try:
                return alg_fn(noisy_audio, sr, clean_audio=clean, **params)
            except TypeError:
                return alg_fn(noisy_audio, sr, **params)
        return alg_fn(noisy_audio, sr, **params)

    opt = optimize_parameters(clean, noisy, sr, algorithm_wrapper, param_ranges)

    enhanced_stoi = opt["stoi"]["enhanced"]
    enhanced_pesq = opt["pesq"]["enhanced"]
    enhanced_bal = opt["balance"]["enhanced"]

    os.makedirs(out_dir, exist_ok=True)

    path_stoi = os.path.join(out_dir, f"{stem}_{alg_name}_optimized_stoi.wav")
    path_pesq = os.path.join(out_dir, f"{stem}_{alg_name}_optimized_pesq.wav")
    path_bal = os.path.join(out_dir, f"{stem}_{alg_name}_optimized_balanced.wav")

    sf.write(path_stoi, enhanced_stoi, sr)
    sf.write(path_pesq, enhanced_pesq, sr)
    sf.write(path_bal, enhanced_bal, sr)

    return {
        "alg": alg_name,
        "stem": stem,
        "sr": sr,

        "stoi_noisy": opt["baseline"]["stoi"],
        "pesq_noisy": opt["baseline"]["pesq"],
        "snr_noisy": opt["baseline"]["snr"],

        "stoi_stoiopt": opt["stoi"]["score"],
        "pesq_stoiopt": opt["stoi"]["pesq"],
        "snr_stoiopt": opt["stoi"]["snr"],

        "stoi_pesqopt": opt["pesq"]["stoi"],
        "pesq_pesqopt": opt["pesq"]["score"],
        "snr_pesqopt": opt["pesq"]["snr"],

        "stoi_balopt": opt["balance"]["stoi"],
        "pesq_balopt": opt["balance"]["pesq"],
        "snr_balopt": opt["balance"]["snr"],

        "best_params_stoi": opt["stoi"].get("params", {}),
        "best_params_pesq": opt["pesq"].get("params", {}),
        "best_params_balanced": opt["balance"].get("params", {})
    }


def _compute_and_save_summary(all_results, algorithms, summary_dir):
    print("\nComputing summary from existing results...")
    summary = {}

    for alg_name, _, _, _ in algorithms:
        rows = [r for r in all_results if r["alg"] == alg_name]

        def safe_mean(values):
            valid = [v for v in values if v is not None]
            return float(np.mean(valid)) if valid else None

        summary[alg_name] = {
            "count": len(rows),
            "stoi_noisy_mean": safe_mean([r["stoi_noisy"] for r in rows]),
            "pesq_noisy_mean": safe_mean([r["pesq_noisy"] for r in rows]),

            "stoi_stoiopt_mean": safe_mean([r["stoi_stoiopt"] for r in rows]),
            "pesq_stoiopt_mean": safe_mean([r["pesq_stoiopt"] for r in rows]),

            "stoi_pesqopt_mean": safe_mean([r["stoi_pesqopt"] for r in rows]),
            "pesq_pesqopt_mean": safe_mean([r["pesq_pesqopt"] for r in rows]),

            "stoi_balopt_mean": safe_mean([r.get("stoi_balopt") for r in rows]),
            "pesq_balopt_mean": safe_mean([r.get("pesq_balopt") for r in rows]),
            "snr_balopt_mean": safe_mean([r.get("snr_balopt") for r in rows])
        }

    summary_path = os.path.join(summary_dir, "summary_means.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser(description='PRECISE batch comparison')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start-from', type=str, default='')
    parser.add_argument('--list-processed', action='store_true')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    summary_dir = os.path.join(base_dir, "results_summary")
    _ensure_dir(summary_dir)

    from spectral_subtractor import spectral_subtraction
    from mmse import mmse
    from wiener_filter import wiener_filter
    from advanced_mmse import advanced_mmse

    algorithms = [
        ("spectralSubtractor", spectral_subtraction, param_ranges_ss,
         os.path.join(base_dir, "results_spectralSubtractor")),
        ("mmse", mmse, param_ranges_mmse, os.path.join(base_dir, "results_mmse")),
        ("wiener", wiener_filter, param_ranges_wiener, os.path.join(base_dir, "results_wiener")),
        ("omlsa", advanced_mmse, param_ranges_omlsa, os.path.join(base_dir, "results_omlsa")),
    ]

    pairs = _find_pairs(data_dir)
    target_sr = 16000

    def get_processed_stems():
        processed_file = set()
        for _, _, _, d in algorithms:
            if os.path.exists(d):
                for file in os.listdir(d):
                    if '_stoi.wav' in file:
                        parts = file.split('_')
                        if len(parts) >= 2: processed_file.add('_'.join(parts[:2]))
        return processed_file

    if args.list_processed:
        for stem in sorted(get_processed_stems()): print(f" {stem}")
        return

    if args.resume or args.start_from:
        print("\n" + "=" * 60 + "\nResume mode\n" + "=" * 60)
        processed_stems = get_processed_stems()
        original_count = len(pairs)
        if args.resume: pairs = [p for p in pairs if p["stem"] not in processed_stems]
        if args.start_from:
            idx = 0
            for i, p in enumerate(pairs):
                if p["stem"] == args.start_from: idx = i; break
            pairs = pairs[idx:]
        print(f"Remaining: {len(pairs)}/{original_count}")
        if len(pairs) == 0 or input("\nContinue? (y/n): ").lower() != 'y': return

    all_results = []
    json_path = os.path.join(summary_dir, "all_results.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)

    for i, p in enumerate(pairs, 1):
        stem = p["stem"]
        print(f"\n[{i}/{len(pairs)}] Processing: {stem}")
        clean, sr_c = librosa.load(p["clean"], sr=None)
        noisy, sr_n = librosa.load(p["noisy"], sr=None)
        clean = _resample_if_needed(clean, sr_c, target_sr)
        noisy = _resample_if_needed(noisy, sr_n, target_sr)
        L = min(len(clean), len(noisy))
        clean, noisy = clean[:L], noisy[:L]

        for alg_name, alg_fn, ranges, out_dir in algorithms:
            # Check if exists in JSON
            if any(r.get("stem") == stem and r.get("alg") == alg_name for r in all_results):
                continue

            res = run_algorithm_on_pair(alg_name, alg_fn, ranges, clean, noisy, target_sr, out_dir, stem)
            all_results.append(res)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    _compute_and_save_summary(all_results, algorithms, summary_dir)

    csv_path = os.path.join(summary_dir, "all_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = ["stem", "alg", "stoi_noisy", "pesq_noisy", "stoi_stoiopt", "pesq_stoiopt", "stoi_pesqopt",
                  "pesq_pesqopt", "stoi_balopt", "pesq_balopt", "snr_balopt"]
        f.write(",".join(header) + "\n")
        for r in all_results:
            row = [r["stem"], r["alg"], _fmt(r["stoi_noisy"]), _fmt(r["pesq_noisy"]), _fmt(r["stoi_stoiopt"]),
                   _fmt(r["pesq_stoiopt"]), _fmt(r["stoi_pesqopt"]), _fmt(r["pesq_pesqopt"]),
                   _fmt(r.get("stoi_balopt")), _fmt(r.get("pesq_balopt")), _fmt(r.get("snr_balopt"))]
            f.write(",".join(row) + "\n")

    print(f"\nFinished - Results in {summary_dir}")


if __name__ == "__main__":
    main()