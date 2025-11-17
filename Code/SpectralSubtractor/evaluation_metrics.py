import numpy as np
import librosa
from pystoi import stoi
import pesq

# STOI – Speech intelligibility, 0–1, how well speech can be understood
# PESQ – Speech quality, -0.5–4.5, how natural it sounds
# ViSQOL – Speech quality, -0.5–4.5, better suited for modern audio processing

def calculate_pesq(clean_reference, test_audio, sr):
    try:
        # Adjust signals to same length
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
            clean_pesq = librosa.resample(clean_trimmed, orig_sr=sr, target_sr=16000)
            test_pesq = librosa.resample(test_trimmed, orig_sr=sr, target_sr=16000)
            mode = 'wb'
            sr_pesq = 16000
            clean_trimmed, test_trimmed = clean_pesq, test_pesq

        # Calculate PESQ score
        pesq_score = pesq.pesq(sr_pesq, clean_trimmed, test_trimmed, mode)
        return pesq_score

    except Exception as e:
        print(f"PESQ calculation failed: {e}")
        return None


def calculate_stoi(clean_reference, test_audio, sr):
    try:
        min_length = min(len(clean_reference), len(test_audio))
        stoi_score = stoi(clean_reference[:min_length], test_audio[:min_length], sr, extended=False)
        return stoi_score
    except Exception as e:
        print(f"STOI calculation failed: {e}")
        return None


def evaluate_audio_quality(clean_reference, noisy_audio, enhanced_audio, sr, algorithm_name=""):

    results = {}

    # STOI Evaluation
    stoi_noisy = calculate_stoi(clean_reference, noisy_audio, sr)
    stoi_enhanced = calculate_stoi(clean_reference, enhanced_audio, sr)

    if stoi_noisy is not None and stoi_enhanced is not None:
        results['stoi_noisy'] = stoi_noisy
        results['stoi_enhanced'] = stoi_enhanced
        results['stoi_improvement'] = stoi_enhanced - stoi_noisy
        results['stoi_improvement_percent'] = (results['stoi_improvement'] / stoi_noisy) * 100

    # PESQ Evaluation
    pesq_noisy = calculate_pesq(clean_reference, noisy_audio, sr)
    pesq_enhanced = calculate_pesq(clean_reference, enhanced_audio, sr)

    if pesq_noisy is not None and pesq_enhanced is not None:
        results['pesq_noisy'] = pesq_noisy
        results['pesq_enhanced'] = pesq_enhanced
        results['pesq_improvement'] = pesq_enhanced - pesq_noisy

    # Print results
    if algorithm_name:
        print(f"\n{'=' * 60}")
        print(f"Evaluation: {algorithm_name}")
        print(f"{'=' * 60}")

    if 'stoi_noisy' in results:
        print(
            f"STOI: {results['stoi_noisy']:.4f} -> {results['stoi_enhanced']:.4f} (+{results['stoi_improvement']:.4f})")

    if 'pesq_noisy' in results:
        print(
            f"PESQ: {results['pesq_noisy']:.2f} -> {results['pesq_enhanced']:.2f} (+{results['pesq_improvement']:.2f})")

    return results


def optimize_parameters(clean_reference, noisy_audio, sr, algorithm_function, param_ranges):
    print(f"\n{'=' * 60}")
    print("Parameter Optimization")
    print(f"{'=' * 60}")

    best_stoi = 0
    best_pesq = 0
    best_params_stoi = {}
    best_params_pesq = {}
    best_enhanced_stoi = None
    best_enhanced_pesq = None

    total_combinations = np.prod([len(v) for v in param_ranges.values()])
    current_combination = 0

    print(f"Testing {total_combinations} parameter combinations")

    # Generate all parameter combinations
    from itertools import product
    param_combinations = product(*param_ranges.values())
    param_names = list(param_ranges.keys())

    for params in param_combinations:
        current_combination += 1
        param_dict = dict(zip(param_names, params))

        # Apply algorithm with current parameters
        enhanced = algorithm_function(noisy_audio, sr, **param_dict)

        # Evaluate STOI
        stoi_score = calculate_stoi(clean_reference, enhanced, sr)
        if stoi_score and stoi_score > best_stoi:
            best_stoi = stoi_score
            best_params_stoi = param_dict.copy()
            best_enhanced_stoi = enhanced

        # Evaluate PESQ
        pesq_score = calculate_pesq(clean_reference, enhanced, sr)
        if pesq_score and pesq_score > best_pesq:
            best_pesq = pesq_score
            best_params_pesq = param_dict.copy()
            best_enhanced_pesq = enhanced

        # Progress indicator
        if current_combination % 10 == 0:
            print(f"  Progress: {current_combination}/{total_combinations}")

    # Display results
    print(f"\nOptimal Parameters for stoi:")
    print(f"  {best_params_stoi}")
    print(f"  STOI: {best_stoi:.4f}")

    print(f"\nOptimal Parameters for pesq:")
    print(f"  {best_params_pesq}")
    print(f"  PESQ: {best_pesq:.2f}")

    print("=" * 60)

    return {
        'stoi': {'enhanced': best_enhanced_stoi, 'params': best_params_stoi, 'score': best_stoi},
        'pesq': {'enhanced': best_enhanced_pesq, 'params': best_params_pesq, 'score': best_pesq}
    }