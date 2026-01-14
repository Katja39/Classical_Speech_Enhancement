import os
import soundfile as sf

import numpy as np
import librosa

from evaluation_metrics import evaluate_audio_quality
from speech_enhancement_comparison import optimize_parameters

from load_files import load_clean_noisy, default_out_dir

from noise_estimation import noise_estimation

from parameter_ranges import param_ranges_ss

def spectral_subtraction(noisy_audio, sr, alpha, beta, n_fft, hop_length, noise_percentile, noise_method, clean_audio=None):

    noisy_audio = np.asarray(noisy_audio, dtype=np.float64)
    original_length = len(noisy_audio)

    # STFT
    stft_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    power_noisy = np.abs(stft_noisy) ** 2

    # noise estimation
    power_noise = noise_estimation(
        noisy_audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        percentile=noise_percentile,
        method=noise_method,
        clean_audio=clean_audio,
        eps=1e-10
    )

    power_clean = power_noisy - alpha * power_noise
    power_clean = np.maximum(power_clean, beta * power_noisy)

    magnitude_clean = np.sqrt(power_clean)
    phase_noisy = np.angle(stft_noisy)
    stft_clean = magnitude_clean * np.exp(1j * phase_noisy)

    clean_audio = librosa.istft(stft_clean, hop_length=hop_length, win_length=n_fft)

    if len(clean_audio) > original_length:
        clean_audio = clean_audio[:original_length]
    elif len(clean_audio) < original_length:
        clean_audio = np.pad(clean_audio, (0, original_length - len(clean_audio)))

    return clean_audio


def test_spectral_subtraction(clean_path=None, noisy_path=None, data_dir="data", out_dir=None, stem="sample"):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if clean_path is None or noisy_path is None:
        data_dir = os.path.join(base_dir, data_dir)
        clean_path = os.path.join(data_dir, "p232_014_clean.wav")
        noisy_path = os.path.join(data_dir, "p232_014_noiseWithMusic.wav")
        stem = "p232_014"

    if out_dir is None:
        out_dir = default_out_dir(base_dir, "results_spectralSubtractor")

    clean_reference, noisy, sr = load_clean_noisy(clean_path, noisy_path, target_sr=16000)

    print("\nStarting Spectral Subtraction parameter optimization...")
    optimization_results = optimize_parameters(
        clean_reference, noisy, sr, spectral_subtraction, param_ranges_ss
    )

    stoi_results = evaluate_audio_quality(clean_reference, noisy, optimization_results['stoi']['enhanced'], sr,
                                         "Spectral Subtraction (STOI optimized)")
    pesq_results = evaluate_audio_quality(clean_reference, noisy, optimization_results['pesq']['enhanced'], sr,
                                         "Spectral Subtraction (PESQ optimized)")

    stoi_path = os.path.join(out_dir, f"{stem}_spectralSubtractor_optimized_stoi.wav")
    pesq_path = os.path.join(out_dir, f"{stem}_spectralSubtractor_optimized_pesq.wav")
    sf.write(stoi_path, optimization_results['stoi']['enhanced'], sr)
    sf.write(pesq_path, optimization_results['pesq']['enhanced'], sr)

    return noisy, optimization_results, sr, stoi_results


def main():
    print("Spectral Subtraction Algorithm")
    print("=" * 60)
    try:
        noisy, optimization_results, sr, results = test_spectral_subtraction()

        if results:
            print("\n" + "=" * 40)
            print("FINAL SUMMARY")
            print("=" * 40)
            print(f"PESQ STOI-optimized: {optimization_results['stoi']['score']:.2f}")
            print(f"PESQ PESQ-optimized: {optimization_results['pesq']['score']:.2f}")
            print("=" * 40)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()