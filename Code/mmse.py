import numpy as np
import librosa
from scipy.special import i0, i1
from noise_estimation import noise_estimation

def mmse(noisy_audio, sr, alpha, ksi_min, gain_min, gain_max, n_fft, hop_length,
         noise_percentile, noise_method, noise_mu=0.98, clean_audio=None,
         log=True, log_every=50):
    """
    Classic Ephraim-Malah MMSE-STSA speech enhancement.
    """
    noisy_audio = np.asarray(noisy_audio, dtype=np.float64)
    if noisy_audio.ndim > 1:
        noisy_audio = np.mean(noisy_audio, axis=1)

    original_length = len(noisy_audio)
    eps = 1e-12

    stft_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=True,
        pad_mode="reflect"
    )

    # 1) STFT analysis: Y is complex STFT of noisy signal
    Y = librosa.stft(noisy_audio, **stft_kwargs)

    # Power spectrum |Y|^2 (shape: n_bins x n_frames)
    power_noisy = np.abs(Y) ** 2
    num_bins, num_frames = Y.shape

    # 2) Noise estimation
    noise_psd_all = noise_estimation(
        noisy_audio, sr=sr,
        method=noise_method,
        n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=stft_kwargs["window"],
        center=stft_kwargs["center"],
        pad_mode=stft_kwargs["pad_mode"],
        percentile=noise_percentile,
        clean_audio=clean_audio,
        eps=eps
    )

    if noise_method != "true_noise" and noise_psd_all.ndim == 2 and noise_psd_all.shape[1] > 1:
        noise_smooth = np.empty_like(noise_psd_all)
        noise_smooth[:, 0] = noise_psd_all[:, 0]
        mu = float(np.clip(noise_mu, 0.0, 0.9999))
        for t in range(1, noise_psd_all.shape[1]):
            noise_smooth[:, t] = mu * noise_smooth[:, t - 1] + (1.0 - mu) * noise_psd_all[:, t]
        noise_psd_all = noise_smooth

    # Detect whether noise PSD is adaptive/time-varying
    is_adaptive = (noise_psd_all.ndim == 2 and noise_psd_all.shape[1] > 1)

    # Gain matrix over all time-frequency bins (same shape as STFT)
    mmse_gain = np.zeros((num_bins, num_frames), dtype=np.float64)

    prev_gain = np.ones((num_bins, 1), dtype=np.float64)
    prev_gamma = np.ones((num_bins, 1), dtype=np.float64)

    for t in range(num_frames):
        # Current noisy power spectrum for frame t
        current_power = power_noisy[:, t:t + 1]

        # Select noise PSD for this frame (adaptive) or static (same for all frames)
        curr_noise = noise_psd_all[:, t:t + 1] if is_adaptive else noise_psd_all
        curr_noise = np.maximum(curr_noise, eps)

        # a-posteriori SNR
        gamma = np.maximum(current_power / curr_noise, eps)

        # Decision-Directed estimation of a-priori SNR (ksi)
        direct = np.maximum(gamma - 1.0, 0.0)
        if t == 0:
            # Initial estimate: ksi = max(gamma - 1, ksi_min)
            ksi = np.maximum(gamma - 1.0, ksi_min)
        else:
            recursive = (prev_gain ** 2) * prev_gamma
            ksi = alpha * recursive + (1.0 - alpha) * direct
            ksi = np.maximum(ksi, ksi_min)

        # v parameter used by Ephraim-Malah gain formula
        # v = (ksi * gamma) / (1 + ksi)
        v = np.clip((ksi * gamma) / (1.0 + ksi), eps, 80.0)

        # Ephraim-Malah MMSE-STSA gain:
        # G = (sqrt(pi)/2) * (sqrt(v)/gamma) * exp(-v/2) * [(1+v)I0(v/2) + v I1(v/2)]
        x = 0.5 * v  # = v/2
        A = (np.sqrt(np.pi) / 2.0) * (np.sqrt(v) / (gamma + eps))
        B = np.exp(-x)
        C = (1.0 + v) * i0(x) + v * i1(x)
        gain = A * B * C

        # Numerical safety
        gain = np.nan_to_num(gain, nan=gain_min, posinf=gain_max, neginf=gain_min)
        gain = np.clip(gain, gain_min, gain_max)

        mmse_gain[:, t:t + 1] = gain

        # Update state for next frame
        prev_gain = gain
        prev_gamma = gamma

    # 4) Synthesis
    stft_enhanced = Y * mmse_gain

    enhanced_audio = librosa.istft(
        stft_enhanced,
        hop_length=stft_kwargs["hop_length"],
        win_length=stft_kwargs["win_length"],
        window=stft_kwargs["window"],
        center=stft_kwargs["center"],
        length=original_length
    )

    return enhanced_audio
