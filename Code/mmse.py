import os
import numpy as np
import librosa
from scipy.special import i0, i1

import soundfile as sf
from evaluation_metrics import evaluate_audio_quality
from speech_enhancement_comparison import optimize_parameters
from load_files import load_clean_noisy, default_out_dir

from noise_estimation import noise_estimation

from parameter_ranges import param_ranges_mmse

def mmse(noisy_audio, sr, alpha, ksi_min, gain_min, gain_max, n_fft, hop_length, noise_percentile, noise_method, clean_audio = None):
    """
    Classic Ephraim-Malah MMSE-STSA speech enhancement

    - Estimates noise power spectrum from a fixed initial segment [noise_start, noise_end]
    - Uses decision-directed approach for a-priori SNR
    - Uses the classical MMSE-STSA gain involving exp() and modified Bessel functions I0/I1
    """

    noisy_audio = np.asarray(noisy_audio, dtype=np.float64)
    original_length = len(noisy_audio)

    eps = 1e-10

    # 1) STFT (noisy)
    stft_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)

    # 2) Noise estimation
    power_noise = noise_estimation(
        noisy_audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        percentile=noise_percentile,
        method=noise_method,
        clean_audio = clean_audio,
        eps=eps
    )

    # 3) Power spectra
    power_noisy = np.abs(stft_noisy) ** 2

    # Guard against zero noise power
    power_noise = np.maximum(power_noise, eps)

    num_freq_bins, num_frames = stft_noisy.shape

    # 4) Allocate
    mmse_gain = np.zeros((num_freq_bins, num_frames), dtype=np.float64)

    prev_gain = None
    prev_power_noisy = None

    # 5) Frame loop
    for t in range(num_frames):
        # a-posteriori SNR γ = |Y|^2 / λ_d
        gamma = power_noisy[:, t:t+1] / power_noise

        if t == 0:
            # ξ(0) = max(γ - 1, ksi_min)
            ksi = np.maximum(gamma - 1.0, ksi_min)
        else:
            # Decision-directed:
            # ξ = α * |Ŝ(m-1)|^2 / λ_d + (1-α) * max(γ - 1, 0)
            # with |Ŝ(m-1)|^2 ≈ G(m-1)^2 * |Y(m-1)|^2
            recursive_part = (prev_gain ** 2) * prev_power_noisy / power_noise
            direct_part = np.maximum(gamma - 1.0, 0.0)
            ksi = alpha * recursive_part + (1.0 - alpha) * direct_part
            ksi = np.maximum(ksi, ksi_min)

        # v = ξ*γ / (1+ξ)
        v = (ksi * gamma) / (1.0 + ksi)

        # Numerical clipping for stability
        v = np.clip(v, 1e-12, 80.0)

        # Classic Ephraim-Malah MMSE-STSA gain:
        # G = (sqrt(pi)/2) * sqrt(v)/γ * exp(-v/2) * [(1+v)I0(v/2) + v I1(v/2)]
        # Note: gamma is the a-posteriori SNR (γ)
        exp_term = np.exp(-0.5 * v)
        i0_term = i0(0.5 * v)
        i1_term = i1(0.5 * v)

        bessel_part = (1.0 + v) * i0_term + v * i1_term
        gain = (np.sqrt(np.pi) / 2.0) * (np.sqrt(v) * exp_term * bessel_part) / (gamma + eps)

        gain = np.nan_to_num(gain, nan=0.0, posinf=gain_max, neginf=0.0)
        gain = np.clip(gain, gain_min, gain_max)

        mmse_gain[:, t:t+1] = gain

        # Update prev for next frame
        prev_gain = gain
        prev_power_noisy = power_noisy[:, t:t+1]

    # 6) Apply gain and ISTFT
    stft_clean = stft_noisy * mmse_gain
    clean_audio = librosa.istft(stft_clean, hop_length=hop_length, win_length=n_fft)

    # 7) Length fix
    if len(clean_audio) > original_length:
        clean_audio = clean_audio[:original_length]
    elif len(clean_audio) < original_length:
        clean_audio = np.pad(clean_audio, (0, original_length - len(clean_audio)))

    return clean_audio

def test_mmse(clean_path=None, noisy_path=None, data_dir="data", out_dir=None, stem="sample"):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if clean_path is None or noisy_path is None:
        data_dir = os.path.join(base_dir, data_dir)
        clean_path = os.path.join(data_dir, "p232_014_clean.wav")
        noisy_path = os.path.join(data_dir, "p232_014_noiseWithMusic.wav")
        stem = "p232_014"

    if out_dir is None:
        out_dir = default_out_dir(base_dir, "results_mmse")

    clean_reference, noisy, sr = load_clean_noisy(clean_path, noisy_path, target_sr=16000)

    print("\nStarting MMSE parameter optimization...")
    optimization_results = optimize_parameters(
        clean_reference, noisy, sr, mmse, param_ranges_mmse
    )

    # Evaluate both optimized results
    stoi_results = evaluate_audio_quality(clean_reference, noisy, optimization_results['stoi']['enhanced'], sr,
                                         "MMSE (STOI optimized)")
    pesq_results = evaluate_audio_quality(clean_reference, noisy, optimization_results['pesq']['enhanced'], sr,
                                         "MMSE (PESQ optimized)")

    # Save
    stoi_path = os.path.join(out_dir, f"{stem}_mmse_optimized_stoi.wav")
    pesq_path = os.path.join(out_dir, f"{stem}_mmse_optimized_pesq.wav")
    sf.write(stoi_path, optimization_results['stoi']['enhanced'], sr)
    sf.write(pesq_path, optimization_results['pesq']['enhanced'], sr)

    return noisy, optimization_results, sr, stoi_results


def main():
    print("MMSE (Minimum Mean Square Error) Speech Enhancement")
    print("=" * 60)
    try:
        noisy, optimization_results, sr, results = test_mmse()

        if results and optimization_results:
            print("\n" + "=" * 40)
            print("FINAL SUMMARY - MMSE")
            print("=" * 40)
            if 'stoi_noisy' in results:
                print(
                    f"STOI: {results['stoi_noisy']:.4f} -> {results['stoi_enhanced']:.4f} (+{results['stoi_improvement']:.4f})")
            print(f"PESQ STOI-optimized: {optimization_results['stoi']['score']:.2f}")
            print(f"PESQ PESQ-optimized: {optimization_results['pesq']['score']:.2f}")

            print("=" * 40)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()