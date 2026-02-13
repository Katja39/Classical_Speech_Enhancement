import librosa
from pystoi import stoi
import numpy as np
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


def calculate_snr(clean, processed):
    try:
        clean = np.asarray(clean)
        processed = np.asarray(processed)
        min_length = min(len(clean), len(processed))
        clean = clean[:min_length]
        processed = processed[:min_length]

        noise = clean - processed
        p_signal = np.sum(clean ** 2)
        p_noise = np.sum(noise ** 2)

        if p_noise == 0:
            return float('inf')

        snr = 10 * np.log10(p_signal / (p_noise + 1e-10))
        return float(snr)
    except Exception as e:
        print(f"SNR calculation failed: {e}")
        return None


def evaluate_audio_quality(clean_reference, noisy_audio, enhanced_audio, sr, algorithm_name=""):
    results = {}

    s_noisy = calculate_stoi(clean_reference, noisy_audio, sr)
    s_enhanced = calculate_stoi(clean_reference, enhanced_audio, sr)

    if s_noisy is not None and s_enhanced is not None:
        results['stoi_noisy'] = s_noisy
        results['stoi_enhanced'] = s_enhanced
        results['stoi_improvement'] = s_enhanced - s_noisy

    p_noisy = calculate_pesq(clean_reference, noisy_audio, sr)
    p_enhanced = calculate_pesq(clean_reference, enhanced_audio, sr)

    if p_noisy is not None and p_enhanced is not None:
        results['pesq_noisy'] = p_noisy
        results['pesq_enhanced'] = p_enhanced
        results['pesq_improvement'] = p_enhanced - p_noisy

    snr_val = calculate_snr(clean_reference, enhanced_audio)
    if snr_val is not None:
        results['snr_enhanced'] = snr_val

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

    if 'snr_enhanced' in results:
        print(f"SNR (Enhanced): {results['snr_enhanced']:.2f} dB")

    return results


def calculate_combined_speech_score(stoi, pesq):
    """
    Calculates a composite score from STOI and PESQ metrics.
    PESQ is divided by 4.5 to scale it to a similar range as STOI (0-1).
    Weighting: 50% STOI, 50% normalized PESQ.
    """
    if stoi is None: stoi = 0
    if pesq is None: pesq = 0

    pesq_norm = max(0, pesq) / 4.5

    return 0.5 * stoi + 0.5 * pesq_norm