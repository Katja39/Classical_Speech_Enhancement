import numpy as np
import librosa

from noise_estimation import noise_estimation

def spectral_subtraction(noisy_audio, sr, alpha, beta, n_fft, hop_length, noise_percentile, noise_method, clean_audio=None):
    """
    Standard Power Spectral Subtraction with over-subtraction (alpha)
    and spectral floor (beta).
    """
    noisy_audio = np.asarray(noisy_audio, dtype=np.float64)
    original_length = len(noisy_audio)

    # 1) STFT analysis
    stft_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    power_noisy = np.abs(stft_noisy) ** 2

    # 2) Noise estimation
    power_noise = noise_estimation(
        noisy_audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, percentile=noise_percentile,
        method=noise_method, clean_audio=clean_audio, eps=1e-10
    )

    # If power_noise is a matrix (TrueNoise), ensure frame alignment
    if power_noise.ndim == 2 and power_noise.shape[1] > 1:
        if power_noise.shape[1] != power_noisy.shape[1]:
            # This is a safety fallback for librosa's frame calculation differences
            power_noise = librosa.util.fix_length(power_noise, size=power_noisy.shape[1], axis=1)

    # 3) Subtraction: D = Y^2 - alpha * N^2
    power_clean = power_noisy - alpha * power_noise

    # 4) Spectral floor: prevent negative values
    # Sets a minimum floor based on the estimated noise level (beta)
    power_clean = np.maximum(power_clean, beta * power_noise)

    # 5) Synthesis
    magnitude_clean = np.sqrt(power_clean)
    # Recombine with original phase
    stft_clean = magnitude_clean * np.exp(1j * np.angle(stft_noisy))

    clean_audio = librosa.istft(stft_clean, hop_length=hop_length, win_length=n_fft)

    # Correct length to match original input
    return librosa.util.fix_length(clean_audio, size=original_length)