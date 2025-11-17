import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pystoi import stoi

import sounddevice as sd

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


def stoi_evaluation(clean_reference, noisy_audio, enhanced_audio, sr):

    # STOI
    # Intrusive metric that estimates speech intelligibility by comparing clean and degraded signals
    # STOI analyzes the spectral structure of speech
    # As long as key speech frequencies (especially 500–4000 Hz) remain identifiable, the score stays high

    try:
        # Adjust all signals to same length
        min_length = min(len(clean_reference), len(noisy_audio), len(enhanced_audio))
        clean_trimmed = clean_reference[:min_length]
        noisy_trimmed = noisy_audio[:min_length]
        enhanced_trimmed = enhanced_audio[:min_length]

        # 1. STOI: Clean vs Noisy (Original quality)
        stoi_noisy = stoi(clean_trimmed, noisy_trimmed, sr, extended=False)

        # 2. STOI: Clean vs Enhanced (Improved quality)
        stoi_enhanced = stoi(clean_trimmed, enhanced_trimmed, sr, extended=False)

        # 3. Calculate improvement
        improvement = stoi_enhanced - stoi_noisy
        improvement_percent = (improvement / stoi_noisy) * 100

        print("\n" + "=" * 60)
        print("Stoi \n")
        print("Comparison with clean reference signal:")
        print(f"  Noisy (Original) vs Clean:    {stoi_noisy:.4f}")
        print(f"  Enhanced (Processed) vs Clean: {stoi_enhanced:.4f}")
        print(f"  Improvement by spectral subtraction algorithm:      +{improvement:.4f} ({improvement_percent:+.1f}%)")

        # Interpretation
        print("\nINTERPRETATION:")
        print(f"  Noisy quality:    {interpret_stoi(stoi_noisy)}")
        print(f"  Enhanced quality: {interpret_stoi(stoi_enhanced)}")

        if improvement > 0:
            print(f"  SUCCESS: Algorithm improves speech intelligibility!")
        else:
            print(f"  PROBLEM: Algorithm reduces quality!")

        print("=" * 60)

        return {
            'stoi_noisy': stoi_noisy,
            'stoi_enhanced': stoi_enhanced,
            'improvement': improvement,
            'improvement_percent': improvement_percent
        }

    except Exception as e:
        print(f"STOI calculation failed: {e}")
        return None


def interpret_stoi(score):
    if score > 0.9:
        return "EXCELLENT"
    elif score > 0.75:
        return "GOOD"
    elif score > 0.6:
        return "OK"
    else:
        return "POOR"


def test_algorithm():
    # Load test signals
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading audio files")

    # 1. Load noisy signal
    noisy, sr = librosa.load("p232_014_noiseWithMusic.wav", sr=16000)
    print(f"Noisy audio loaded: {len(noisy)} samples, {sr} Hz")

    # 2. Load clean reference
    clean_reference, sr_clean = librosa.load("p232_014_clean.wav", sr=16000)
    print(f"Clean reference loaded: {len(clean_reference)} samples, {sr_clean} Hz")

    # Check sampling rates
    if sr != sr_clean:
        print(f"Warning: Different sampling rates - Noisy: {sr}Hz, Clean: {sr_clean}Hz")
        # Resample if necessary
        if sr_clean != 16000:
            clean_reference = librosa.resample(clean_reference, orig_sr=sr_clean, target_sr=sr)

    # 3. Apply Spectral Subtraction
    print("Applying Spectral Subtraction...")
    enhanced = spectral_subtraction(noisy, sr)
    print(f"Enhanced audio created: {len(enhanced)} samples")

    # 4. Save the denoised signal
    denoised_path = os.path.join(current_dir, "p232_014_noiseWithMusic_denoised.wav")
    import soundfile as sf
    sf.write(denoised_path, enhanced, sr)
    print(f"Denoised audio saved: {denoised_path}")

    # 5. QUALITY ASSESSMENT WITH CLEAN REFERENCE
    results = stoi_evaluation(clean_reference, noisy, enhanced, sr)

    # 6. Audio playback (optional)
    #try:
    #    print("\nAudio comparison (optional):")
    #    print("Playing Noisy signal...")
    #    sd.play(noisy, sr)
    #    sd.wait()
#
    #    print("Playing Enhanced signal...")
    #    sd.play(enhanced, sr)
    #    sd.wait()
#
    #    print("Playing Clean Reference...")
    #    sd.play(clean_reference, sr)
    #    sd.wait()
    #except Exception as e:
    #    print(f"Audio playback not available: {e}")

    return noisy, enhanced, sr, results


def main():
    print("Starting Spectral Subtraction with QUALITY EVALUATION")
    print("=" * 60)
    try:
        noisy, enhanced, sr, results = test_algorithm()

        if results:
            print(f"\nSUMMARY:")
            print(f"Initial quality (Noisy): {results['stoi_noisy']:.4f}")
            print(f"Final quality (Enhanced): {results['stoi_enhanced']:.4f}")
            print(f"Quality improvement: {results['improvement']:.4f} ({results['improvement_percent']:+.1f}%)")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()