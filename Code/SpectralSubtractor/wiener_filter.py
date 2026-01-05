import os
import soundfile as sf

import numpy as np
import librosa

from evaluation_metrics import evaluate_audio_quality, optimize_parameters
from load_files import load_clean_noisy, default_out_dir

def wiener_filter(noisy_audio, sr,
                  noise_start=0.0, noise_end=0.1,
                  n_fft=1024, hop_length=256,
                  alpha=0.98,
                  gain_floor=0.001):
    """
    Classic Wiener filtering (single-channel) in STFT domain.

    Steps:
    1) Estimate noise PSD from initial segment
    2) Compute a-posteriori SNR gamma = |Y|^2 / lambda_d
    3) Estimate a-priori SNR ksi via Decision-Directed (optional but common)
    4) Wiener gain: G = ksi / (1 + ksi)
    5) Apply gain to noisy STFT (keep noisy phase)
    6) ISTFT + length fix
    """

    noisy_audio = np.asarray(noisy_audio, dtype=np.float64)
    original_length = len(noisy_audio)
    eps = 1e-10

    # 1) Noise estimation from initial segment
    ns = int(max(0.0, noise_start) * sr)
    ne = int(max(0.0, noise_end) * sr)
    ne = min(ne, original_length)
    if ne <= ns:
        ns, ne = 0, min(int(0.1 * sr), original_length)

    noise_segment = noisy_audio[ns:ne]
    if len(noise_segment) < 2:
        noise_segment = noisy_audio[:min(original_length, max(2, int(0.1 * sr)))]

    # 2) STFT
    stft_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    stft_noise = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)

    # 3) Power spectra
    power_noisy = np.abs(stft_noisy) ** 2
    power_noise = np.mean(np.abs(stft_noise) ** 2, axis=1, keepdims=True)
    power_noise = np.maximum(power_noise, eps)

    num_freq_bins, num_frames = stft_noisy.shape

    # 4) Decision-Directed a-priori SNR (ksi) and Wiener gain
    wiener_gain = np.zeros((num_freq_bins, num_frames), dtype=np.float64)

    prev_gain = None
    prev_power_noisy = None

    for t in range(num_frames):
        # a-posteriori SNR
        gamma = power_noisy[:, t:t+1] / power_noise

        if t == 0:
            # initial a-priori SNR
            ksi = np.maximum(gamma - 1.0, 0.0)
        else:
            # decision-directed estimate
            recursive_part = (prev_gain ** 2) * prev_power_noisy / power_noise
            direct_part = np.maximum(gamma - 1.0, 0.0)
            ksi = alpha * recursive_part + (1.0 - alpha) * direct_part
            ksi = np.maximum(ksi, 0.0)

        # Wiener gain
        gain = ksi / (1.0 + ksi)
        gain = np.clip(gain, gain_floor, 1.0)

        wiener_gain[:, t:t+1] = gain

        prev_gain = gain
        prev_power_noisy = power_noisy[:, t:t+1]

    # 5) Apply gain (keep noisy phase implicitly by complex multiplication)
    stft_clean = stft_noisy * wiener_gain

    # 6) ISTFT
    clean_audio = librosa.istft(stft_clean, hop_length=hop_length, win_length=n_fft)

    # Length fix
    if len(clean_audio) > original_length:
        clean_audio = clean_audio[:original_length]
    elif len(clean_audio) < original_length:
        clean_audio = np.pad(clean_audio, (0, original_length - len(clean_audio)))

    return clean_audio


def test_wiener(clean_path=None, noisy_path=None, data_dir="data", out_dir=None, stem="sample"):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if clean_path is None or noisy_path is None:
        data_dir = os.path.join(base_dir, data_dir)
        clean_path = os.path.join(data_dir, "p232_014_clean.wav")
        noisy_path = os.path.join(data_dir, "p232_014_noiseWithMusic.wav")
        stem = "p232_014"

    if out_dir is None:
        out_dir = default_out_dir(base_dir, "results_wiener")

    clean_reference, noisy, sr = load_clean_noisy(clean_path, noisy_path, target_sr=16000)

    param_ranges = {
        "alpha": [0.85, 0.9, 0.94, 0.96, 0.98],#0.92
        "n_fft": [512, 1024],
        "hop_length": [128, 256],
        "gain_floor": [0.0005, 0.001, 0.005, 0.01],
    }

    print("\nStarting Wiener parameter optimization...")
    optimization_results = optimize_parameters(clean_reference, noisy, sr, wiener_filter, param_ranges)

    stoi_results = evaluate_audio_quality(clean_reference, noisy, optimization_results['stoi']['enhanced'], sr,
                                         "Wiener (STOI optimized)")
    pesq_results = evaluate_audio_quality(clean_reference, noisy, optimization_results['pesq']['enhanced'], sr,
                                         "Wiener (PESQ optimized)")

    stoi_path = os.path.join(out_dir, f"{stem}_wiener_optimized_stoi.wav")
    pesq_path = os.path.join(out_dir, f"{stem}_wiener_optimized_pesq.wav")
    sf.write(stoi_path, optimization_results['stoi']['enhanced'], sr)
    sf.write(pesq_path, optimization_results['pesq']['enhanced'], sr)

    return noisy, optimization_results, sr, stoi_results


def main():
    print("Wiener Filter Speech Enhancement")
    print("=" * 60)
    try:
        noisy, optimization_results, sr, results = test_wiener()

        if results and optimization_results:
            print("\n" + "=" * 40)
            print("FINAL SUMMARY - WIENER")
            print("=" * 40)
            if "stoi_noisy" in results:
                print(f"STOI: {results['stoi_noisy']:.4f} -> {results['stoi_enhanced']:.4f} (+{results['stoi_improvement']:.4f})")
            print(f"PESQ STOI-optimized: {optimization_results['stoi']['score']:.2f}")
            print(f"PESQ PESQ-optimized: {optimization_results['pesq']['score']:.2f}")
            print("=" * 40)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()