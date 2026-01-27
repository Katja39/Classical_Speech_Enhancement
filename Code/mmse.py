import numpy as np
import librosa
from scipy.special import i0, i1

from noise_estimation import noise_estimation

def mmse(noisy_audio, sr, alpha, ksi_min, gain_min, gain_max, n_fft, hop_length,
         noise_percentile, noise_method, noise_mu=0.98, clean_audio=None):
    """
     Classic Ephraim-Malah MMSE-STSA speech enhancement
    """

    noisy_audio = np.asarray(noisy_audio, dtype=np.float64)
    original_length = len(noisy_audio)
    eps = 1e-10

    # 1) STFT analysis
    Y = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    power_noisy = np.abs(Y) ** 2
    num_bins, num_frames = Y.shape

    # 2) Noise estimation (Returns vector or matrix)
    noise_psd_all = noise_estimation(
        noisy_audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, percentile=noise_percentile,
        method=noise_method, clean_audio=clean_audio, eps=eps
    )

    # Check if noise is adaptive (Matrix) or stationary (Vector)
    is_adaptive = (noise_psd_all.shape[1] > 1)

    mmse_gain = np.zeros((num_bins, num_frames), dtype=np.float64)
    prev_gain = np.ones((num_bins, 1))
    prev_power_noisy = power_noisy[:, 0:1]

    # Internal noise profile for recursive updates (only used if not is_adaptive)
    active_noise_psd = noise_psd_all[:, 0:1] if is_adaptive else noise_psd_all

    # 3) Frame processing loop
    for t in range(num_frames):
        current_power = power_noisy[:, t:t + 1]

        # Select current noise profile
        curr_noise = noise_psd_all[:, t:t + 1] if is_adaptive else active_noise_psd

        # a-posteriori SNR (gamma)
        gamma = np.maximum(current_power / curr_noise, eps)

        # Decision-Directed a-priori SNR (ksi)
        recursive = (prev_gain ** 2) * prev_power_noisy / curr_noise
        direct = np.maximum(gamma - 1.0, 0.0)

        if t == 0:
            ksi = np.maximum(gamma - 1.0, ksi_min)
        else:
            ksi = alpha * recursive + (1.0 - alpha) * direct
            ksi = np.maximum(ksi, ksi_min)

        # Helper variable v for the gain formula
        v = np.clip((ksi * gamma) / (1.0 + ksi), 1e-10, 80.0)

        # Ephraim-Malah MMSE-STSA Gain Function:
        # G = (sqrt(pi)/2) * (sqrt(v)/gamma) * exp(-v/2) * [(1+v)I0(v/2) + vI1(v/2)]
        A = (np.sqrt(np.pi) / 2.0) * (np.sqrt(v) / (gamma + eps))
        B = np.exp(-0.5 * v)
        C = (1.0 + v) * i0(0.5 * v) + v * i1(0.5 * v)

        gain = A * B * C

        # Guard against numerical instability
        gain = np.nan_to_num(gain, nan=gain_min, posinf=gain_max, neginf=gain_min)
        gain = np.clip(gain, gain_min, gain_max)
        mmse_gain[:, t:t + 1] = gain

        if not is_adaptive:
            # Simple VAD-less noise update based on speech probability proxy
            p_speech = np.clip(ksi / (1.0 + ksi), 0.1, 0.9)
            p_noise = 1.0 - p_speech
            actual_noise_est = p_noise * current_power + p_speech * active_noise_psd
            active_noise_psd = noise_mu * active_noise_psd + (1.0 - noise_mu) * actual_noise_est
            active_noise_psd = np.maximum(active_noise_psd, eps)

        prev_gain = gain
        prev_power_noisy = current_power

    # 4) Synthesis
    stft_enhanced = Y * mmse_gain
    enhanced_audio = librosa.istft(stft_enhanced, hop_length=hop_length, win_length=n_fft)

    return librosa.util.fix_length(enhanced_audio, size=original_length)