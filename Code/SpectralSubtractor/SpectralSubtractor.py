import os

import numpy as np
import librosa
import matplotlib.pyplot as plt


def simple_spectral_subtraction(noisy_audio, sr, noise_start=0, noise_end=0.1, alpha=2.0, beta=0.01):
    # 1. Noise estimation from first 100ms
    noise_segment = noisy_audio[int(noise_start * sr):int(noise_end * sr)]

    # 2. STFT - Short-Time Fourier Transform
    stft_noisy = librosa.stft(noisy_audio, n_fft=1024, hop_length=256, win_length=1024)
    stft_noise = librosa.stft(noise_segment, n_fft=1024, hop_length=256, win_length=1024)

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

    clean_audio = librosa.istft(stft_clean, hop_length=256, win_length=1024)

    return clean_audio
    pass


def test_basic_function():
    # Load a test signal
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(current_dir, "p232_014_noiseWithMusic.wav")
    print(f"Searching for audio file at: {audio_path}")

    noisy, sr = librosa.load("p232_014_noiseWithMusic.wav", sr=16000)
    print(f"Audio successfully loaded: {len(noisy)} samples, {sr} Hz")

    # Apply spectral subtraction
    clean = simple_spectral_subtraction(noisy, sr)

    # Save the denoised signal
    denoised_path = os.path.join(current_dir, "p232_014_noiseWithMusic_denoised.wav")
    import soundfile as sf
    sf.write(denoised_path, clean, sr)
    print(f"Denoised audio saved as: {denoised_path}")

    # Listen to the results
    #import sounddevice as sd
    #print("Playing noisy signal...")
    #sd.play(noisy, sr)
    #sd.wait()

    print("Playing denoised signal...")
    sd.play(clean, sr)
    sd.wait()

    return noisy, clean, sr

def main():
    print("Starting spectral subtraction...")
    try:
        noisy, clean, sr = test_basic_function()
        print("Successfully completed!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()