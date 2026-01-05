import os
import numpy as np
import librosa
from evaluation_metrics import calculate_pesq, calculate_stoi, evaluate_audio_quality, optimize_parameters


def spectral_subtraction(noisy_audio, sr, noise_start=0, noise_end=0.1,
                         alpha=2.0, beta=0.01, n_fft=1024, hop_length=256):
    # Store original length
    original_length = len(noisy_audio)

    # 1. Noise estimation from first 100ms
    noise_segment = noisy_audio[int(noise_start * sr):int(noise_end * sr)]

    # 2. STFT - Short-Time Fourier Transform
    stft_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    stft_noise = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)

    # 3. Calculate power spectra
    power_noisy = np.abs(stft_noisy) ** 2
    power_noise = np.mean(np.abs(stft_noise) ** 2, axis=1, keepdims=True)

    # 4. Spectral subtraction
    power_clean = power_noisy - alpha * power_noise
    power_clean = np.maximum(power_clean, beta * power_noisy)

    # 5. Inverse transformation
    magnitude_clean = np.sqrt(power_clean)
    phase_noisy = np.angle(stft_noisy)
    stft_clean = magnitude_clean * np.exp(1j * phase_noisy)

    clean_audio = librosa.istft(stft_clean, hop_length=hop_length, win_length=n_fft)

    # Adjust to original length
    if len(clean_audio) > original_length:
        clean_audio = clean_audio[:original_length]
    elif len(clean_audio) < original_length:
        clean_audio = np.pad(clean_audio, (0, original_length - len(clean_audio)))

    return clean_audio


def test_spectral_subtraction():
    # Load test signals
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading audio files for Spectral Subtraction")

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

    # Define parameter ranges for optimization
    param_ranges = {
        'alpha': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        'beta': [0.001, 0.005, 0.01, 0.02, 0.05],
        'n_fft': [512, 1024]
    }

    # Optimize parameters
    optimization_results = optimize_parameters(
        clean_reference, noisy, sr, spectral_subtraction, param_ranges
    )

    # Evaluate both optimized results
    stoi_results = evaluate_audio_quality(
        clean_reference, noisy, optimization_results['stoi']['enhanced'], sr,
        "Spectral Subtraction (STOI optimized)"
    )

    pesq_results = evaluate_audio_quality(
        clean_reference, noisy, optimization_results['pesq']['enhanced'], sr,
        "Spectral Subtraction (PESQ optimized)"
    )

    # Save both optimized denoised signals
    import soundfile as sf
    denoised_stoi_path = os.path.join(current_dir, "p232_014_spectralSubtractor_denoised_optimized_stoi.wav")
    denoised_pesq_path = os.path.join(current_dir, "p232_014_spectralSubtractor_denoised_optimized_pesq.wav")

    sf.write(denoised_stoi_path, optimization_results['stoi']['enhanced'], sr)
    sf.write(denoised_pesq_path, optimization_results['pesq']['enhanced'], sr)

    print(f"\nOptimized denoised audio saved:")
    print(f"  STOI optimized: {denoised_stoi_path}")
    print(f"  PESQ optimized: {denoised_pesq_path}")

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