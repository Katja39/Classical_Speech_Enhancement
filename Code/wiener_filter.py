
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
    original_length = len(noisy_audio)
    eps = 1e-10

    stft_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    power_noisy = np.abs(stft_noisy) ** 2
    num_freq_bins, num_frames = stft_noisy.shape

    noise_psd_all = noise_estimation(
        noisy_audio, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        percentile=noise_percentile, method=noise_method, clean_audio=clean_audio, eps=eps
    )

    is_adaptive = (noise_psd_all.shape[1] > 1)
    wiener_gain = np.zeros((num_freq_bins, num_frames), dtype=np.float64)

    prev_gain = np.ones((num_freq_bins, 1)) * gain_floor
    prev_power_noisy = power_noisy[:, 0:1]


    for t in range(num_frames):
        curr_noise = noise_psd_all[:, t:t + 1] if is_adaptive else noise_psd_all
        gamma = np.maximum(power_noisy[:, t:t + 1] / curr_noise, eps)

        # Decision-Directed A-priori SNR
        if t == 0:
            ksi = np.maximum(gamma - 1.0, 0.0)
        else:
            recursive_part = (prev_gain ** 2) * prev_power_noisy / curr_noise
            direct_part = np.maximum(gamma - 1.0, 0.0)
            ksi = alpha * recursive_part + (1.0 - alpha) * direct_part

        ksi = np.maximum(ksi, 1e-10)


        gain = np.clip(ksi / (1.0 + ksi), gain_floor, 1.0)
        wiener_gain[:, t:t + 1] = gain

        prev_gain = gain
        prev_power_noisy = power_noisy[:, t:t + 1]

    stft_clean = stft_noisy * wiener_gain
    enhanced = librosa.istft(stft_clean, hop_length=hop_length, win_length=n_fft)
    return librosa.util.fix_length(enhanced, size=original_length)