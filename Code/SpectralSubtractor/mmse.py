import os
import numpy as np
import librosa
from scipy.special import i0, i1

from evaluation_metrics import evaluate_audio_quality, optimize_parameters

def mmse(noisy_audio, sr,
         noise_start=0.0, noise_end=0.1,
         n_fft=1024, hop_length=256,
         alpha=0.98, ksi_min=0.001,
         gain_min=0.001, gain_max=1.0,
         v_max=80.0):
    """
    Classic Ephraim-Malah MMSE-STSA speech enhancement

    - Estimates noise power spectrum from a fixed initial segment [noise_start, noise_end]
    - Uses decision-directed approach for a-priori SNR
    - Uses the classical MMSE-STSA gain involving exp() and modified Bessel functions I0/I1
    """

    noisy_audio = np.asarray(noisy_audio, dtype=np.float64)
    original_length = len(noisy_audio)

    eps = 1e-10

    # 1) Noise estimation from initial segment
    ns = int(max(0.0, noise_start) * sr)
    ne = int(max(noise_end, noise_start) * sr)
    ne = min(ne, original_length)
    if ne <= ns:
        # Fallback: if segment invalid, use a short initial chunk
        ns, ne = 0, min(int(0.1 * sr), original_length)

    noise_segment = noisy_audio[ns:ne]

    # 2) STFT
    stft_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    stft_noise = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)

    # 3) Power spectra
    power_noisy = np.abs(stft_noisy) ** 2
    power_noise = np.mean(np.abs(stft_noise) ** 2, axis=1, keepdims=True)  # λ_d(k)

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
        v = np.clip(v, 1e-12, v_max)

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

def test_mmse():
    """Test function for MMSE estimator"""
    # Load test signals
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading audio files for MMSE")

    # Load noisy signal
    noisy, sr = librosa.load("p232_014_noiseWithMusic.wav", sr=16000)
    print(f"Noisy audio loaded: {len(noisy)} samples, {sr} Hz")

    # Load clean reference
    clean_reference, sr_clean = librosa.load("p232_014_clean.wav", sr=16000)
    print(f"Clean reference loaded: {len(clean_reference)} samples, {sr_clean} Hz")

    # Check sampling rates
    if sr != sr_clean:
        print(f"Warning: Different sampling rates - Noisy: {sr}Hz, Clean: {sr_clean}Hz")
        if sr_clean != 16000:
            clean_reference = librosa.resample(clean_reference, orig_sr=sr_clean, target_sr=sr)

    # Define parameter ranges for MMSE optimization
    param_ranges = {
        'alpha': [0.85, 0.9, 0.92, 0.94, 0.96, 0.98],
        'n_fft': [512, 1024],
        'hop_length': [128, 256],
        'ksi_min': [0.001, 0.01, 0.05]
    }

    # Optimize parameters
    print("\nStarting MMSE parameter optimization...")
    optimization_results = optimize_parameters(clean_reference, noisy, sr, mmse, param_ranges)

    # Evaluate both optimized results
    print("\nEvaluating optimized results...")
    stoi_results = evaluate_audio_quality(
        clean_reference, noisy, optimization_results['stoi']['enhanced'], sr,
        "MMSE (STOI optimized)"
    )

    pesq_results = evaluate_audio_quality(
        clean_reference, noisy, optimization_results['pesq']['enhanced'], sr,
        "MMSE (PESQ optimized)"
    )

    # Save both optimized denoised signals
    import soundfile as sf
    denoised_stoi_path = os.path.join(current_dir, "p232_014_mmse_denoised_optimized_stoi.wav")
    denoised_pesq_path = os.path.join(current_dir, "p232_014_mmse_denoised_optimized_pesq.wav")

    sf.write(denoised_stoi_path, optimization_results['stoi']['enhanced'], sr)
    sf.write(denoised_pesq_path, optimization_results['pesq']['enhanced'], sr)

    print(f"\nOptimized denoised audio saved:")
    print(f"  STOI optimized: {denoised_stoi_path}")
    print(f"  PESQ optimized: {denoised_pesq_path}")

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