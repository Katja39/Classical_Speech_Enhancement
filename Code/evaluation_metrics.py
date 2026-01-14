import librosa
from pystoi import stoi
import pesq

# STOI – Speech intelligibility, 0–1, how well speech can be understood
# PESQ – Speech quality, -0.5–4.5, how natural it sounds
# ViSQOL – Speech quality, -0.5–4.5, better suited for modern audio processing

def calculate_pesq(clean_reference, test_audio, sr):
    try:
        min_length = min(len(clean_reference), len(test_audio))
        clean_trimmed = clean_reference[:min_length]
        test_trimmed = test_audio[:min_length]

        if sr == 16000:
            mode = 'wb'
            sr_pesq = 16000
        elif sr == 8000:
            mode = 'nb'
            sr_pesq = 8000
        else:
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
        print(f"STOI: {results['stoi_noisy']:.4f} -> {results['stoi_enhanced']:.4f} (+{results['stoi_improvement']:.4f})")

    if 'pesq_noisy' in results:
        print(f"PESQ: {results['pesq_noisy']:.2f} -> {results['pesq_enhanced']:.2f} (+{results['pesq_improvement']:.2f})")

    return results