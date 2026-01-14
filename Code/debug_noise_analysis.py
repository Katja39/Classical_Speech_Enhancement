import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from noise_estimation import noise_estimation

def analyze_noise_estimation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    noisy_path = os.path.join(data_dir, "p232_014_noiseWithMusic.wav")
    clean_path = os.path.join(data_dir, "p232_014_clean.wav")

    if not os.path.exists(noisy_path):
        print(f"Data not found: {noisy_path}")
        for f in os.listdir(data_dir):
            if "noise" in f.lower() or "noisy" in f.lower():
                noisy_path = os.path.join(data_dir, f)
                print(f"Verwende: {f}")
                break

    # Audio laden
    noisy, sr = librosa.load(noisy_path, sr=16000)
    print(f"Loaded: {len(noisy) / sr:.1f} seconds, SR: {sr}")

    n_fft = 1024
    hop_length = 256

    print("\n1. Original noise_estimation:")
    power_noise = noise_estimation(
        noisy, sr=sr, n_fft=n_fft, hop_length=hop_length,
        percentile=20.0, eps=1e-10, debug=True
    )

    stft = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
    power = np.abs(stft) ** 2

    print(f"\n2. Power Statistics:")
    print(f"   Signal power mean: {np.mean(power):.2e}")
    print(f"   Noise power mean:  {np.mean(power_noise):.2e}")
    print(f"   SNR (estimated):   {10 * np.log10(np.mean(power) / np.mean(power_noise)):.1f} dB")

    frame_energy = np.mean(np.log(power + 1e-10), axis=0)
    print(f"\n3. Frame Energy Analysis:")
    print(f"   Min frame energy:  {np.min(frame_energy):.2f}")
    print(f"   Max frame energy:  {np.max(frame_energy):.2f}")
    print(f"   Mean frame energy: {np.mean(frame_energy):.2f}")
    print(f"   Std frame energy:  {np.std(frame_energy):.2f}")

    threshold = np.percentile(frame_energy, 20)
    quiet_frames = frame_energy <= threshold
    print(
        f"   Quiet frames ({100 * np.sum(quiet_frames) / len(quiet_frames):.0f}%): {np.sum(quiet_frames)}/{len(quiet_frames)}")

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    time = np.arange(len(noisy)) / sr
    plt.plot(time, noisy, alpha=0.7)
    plt.title("Noisy Audio (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    librosa.display.specshow(D, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='hz')
    plt.title("Spectrogram")
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 3)
    frames = np.arange(len(frame_energy))
    plt.plot(frames, frame_energy, 'b-', alpha=0.7, label='Frame Energy')
    plt.axhline(y=threshold, color='r', linestyle='--',
                label=f'20th percentile ({threshold:.2f})')

    quiet_indices = np.where(quiet_frames)[0]
    plt.scatter(quiet_indices, frame_energy[quiet_indices],
                color='green', s=10, label='Quiet frames')

    plt.title("Frame Energy Analysis")
    plt.xlabel("Frame Index")
    plt.ylabel("Log Power")
    plt.legend()
    plt.tight_layout()

    plt.savefig("noise_analysis.png", dpi=150)
    plt.show()

    return noisy, sr, power_noise


def test_simple_enhancement():

    noisy, sr = librosa.load("data/p232_014_noiseWithMusic.wav", sr=16000)
    clean, _ = librosa.load("data/p232_014_clean.wav", sr=16000)

    duration = 5 * sr
    noisy = noisy[:duration]
    clean = clean[:duration]

    from pystoi import stoi
    import pesq

    n_fft = 1024
    hop_length = 256

    stft = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
    power = np.abs(stft) ** 2

    noise_frames = int(0.5 * sr / hop_length)
    power_noise = np.mean(power[:, :noise_frames], axis=1, keepdims=True)
    power_noise = np.maximum(power_noise, 1e-10)

    snr_prior = 10
    power_signal_prior = power_noise * (10 ** (snr_prior / 10))
    gain = power_signal_prior / (power_signal_prior + power_noise)

    gain = np.clip(gain, 0.01, 0.99)

    enhanced_stft = stft * gain
    enhanced = librosa.istft(enhanced_stft, hop_length=hop_length)

    stoi_score = stoi(clean, enhanced, sr, extended=False)
    pesq_score = pesq.pesq(sr, clean, enhanced, 'wb')

    print(f"\nWiener Filter (first 0.5s as noise):")
    print(f"  STOI: {stoi_score:.4f}")
    print(f"  PESQ: {pesq_score:.2f}")

    power_clean = power - 1.5 * power_noise
    power_clean = np.maximum(power_clean, 0.1 * power)

    magnitude_clean = np.sqrt(power_clean)
    stft_clean = magnitude_clean * np.exp(1j * np.angle(stft))
    enhanced_ss = librosa.istft(stft_clean, hop_length=hop_length)

    stoi_ss = stoi(clean, enhanced_ss, sr, extended=False)
    pesq_ss = pesq.pesq(sr, clean, enhanced_ss, 'wb')

    print(f"\nConservative Spectral Subtraction (alpha=1.5, beta=0.1):")
    print(f"  STOI: {stoi_ss:.4f}")
    print(f"  PESQ: {pesq_ss:.2f}")

    sf.write("debug_noisy.wav", noisy, sr)
    sf.write("debug_simple_wiener.wav", enhanced, sr)
    sf.write("debug_simple_ss.wav", enhanced_ss, sr)

    print("\nDebug files saved. Listen to them!")

if __name__ == "__main__":
    print("=" * 70)
    print("NOISE ESTIMATION DEBUGGING")
    print("=" * 70)

    analyze_noise_estimation()

    test_simple_enhancement()