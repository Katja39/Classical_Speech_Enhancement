
import numpy as np
import librosa

from noise_estimation import noise_estimation

def wiener_filter(noisy_audio, sr,
                  n_fft, hop_length,
                  alpha,
                  gain_floor, noise_percentile, noise_method, clean_audio = None):
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
    if noisy_audio.ndim > 1:
        noisy_audio = np.mean(noisy_audio, axis=1)

    original_length = len(noisy_audio)
    eps = 1e-10

    stft_kwargs = dict(
        n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window="hann", center=True, pad_mode="reflect"
    )

    Y = librosa.stft(noisy_audio, **stft_kwargs)

    power_noisy = np.abs(Y) ** 2
    n_bins, n_frames = Y.shape

    noise_psd_all = noise_estimation(
        noisy_audio, sr=sr,
        n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=stft_kwargs["window"], center=stft_kwargs["center"], pad_mode=stft_kwargs["pad_mode"],
        percentile=noise_percentile, method=noise_method,
        clean_audio=clean_audio, eps=eps
    )
    noise_psd_all = np.maximum(noise_psd_all, eps)
    is_adaptive = (noise_psd_all.ndim == 2 and noise_psd_all.shape[1] > 1)

    G = np.zeros((n_bins, n_frames), dtype=np.float64)
    prev_gain = np.ones((n_bins, 1), dtype=np.float64)
    prev_gamma = np.ones((n_bins, 1), dtype=np.float64)
    #prev_power_noisy = power_noisy[:, 0:1]

    for t in range(n_frames):
        #noise profile
        curr_noise = noise_psd_all[:, t:t + 1] if is_adaptive else noise_psd_all
        curr_noise = np.maximum(curr_noise, eps)

        # A-posteriori SNR
        gamma = np.maximum(power_noisy[:, t:t + 1] / curr_noise, eps)

        direct = np.maximum(gamma - 1.0, 0.0)

        # Decision-Directed A-priori SNR
        if t == 0:
            ksi = direct
        else:
            recursive = (prev_gain ** 2) * prev_gamma
            ksi = alpha * recursive + (1.0 - alpha) * direct

        ksi = np.maximum(ksi, 1e-10)

        # Wiener Gain
        gain = ksi / (1.0 + ksi)

        # Clipping
        gain = np.clip(gain, gain_floor, 1.0)

        G[:, t:t + 1] = gain
        prev_gain = gain
        prev_gamma = gamma
        #prev_power_noisy = power_noisy[:, t:t + 1]

    S = Y * G

    enhanced_audio = librosa.istft(
        S,
        hop_length=stft_kwargs["hop_length"],
        win_length=stft_kwargs["win_length"],
        window=stft_kwargs["window"],
        center=stft_kwargs["center"],
        length=original_length
    )
    return enhanced_audio