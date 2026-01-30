import numpy as np
import librosa
from scipy.special import expn  # exponential integral E1(x) = expn(1, x)

from noise_estimation import noise_estimation

def advanced_mmse(noisy_audio, sr, n_fft, hop_length, alpha, ksi_min,
                  q, noise_mu, gain_floor, noise_percentile, noise_method, clean_audio=None,
                  v_max=80.0):
    """
    Ausgangspunkt: MMSE (STSA)

    Combines Log Spectral Amplitude MMSE (LSA-MMSE)
    # with Speech Presence Probability (SPP) for robust noise suppression

    # 1. Log-MMSE criterion: Minimizes mean-square error of log-spectra
    # 2. Decision-Directed SNR estimation: Balances tracking speed and variance
    # 3. SPP-based gain control: Reduces musical noise via soft voice activity detection
    # 4. Adaptive noise PSD update: Continuous noise estimation during speech pauses
    # 5. Exponential integral E1: Provides analytical LSA gain solution
    """

    """ If speech is present → aggressive noise suppression; no speech → more conservative """

    noisy_audio = np.asarray(noisy_audio, dtype=np.float64)
    original_length = len(noisy_audio)
    eps = 1e-10

    # 1. STFT
    Y = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    Y_power = np.abs(Y) ** 2
    num_bins, num_frames = Y.shape

    # 2. Initial noise power spectral density (PSD) estimation
    noise_psd_all = noise_estimation(
        noisy_audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, percentile=noise_percentile,
        method=noise_method, clean_audio=clean_audio, eps=eps
    )

    # Check if noise estimate is adaptive (time-varying) or static
    is_adaptive = (noise_psd_all.shape[1] > 1)

    # Initialize gain matrix (will be applied to STFT)
    G = np.zeros((num_bins, num_frames), dtype=np.float64)
    q_val = float(np.clip(q, 1e-3, 1 - 1e-3))

    # 3. Initialize tracking variables for recursive estimation
    active_noise_psd = noise_psd_all[:, 0:1] if is_adaptive else noise_psd_all

    # Previous gain and power for decision-directed approach
    prev_gain = np.ones((num_bins, 1)) * gain_floor
    prev_Y_power = Y_power[:, 0:1]

    # 4. loop over all time frames
    for t in range(num_frames):
        Yp = Y_power[:, t:t + 1]

        # Select appropriate noise estimate for current frame
        curr_noise = noise_psd_all[:, t:t + 1] if is_adaptive else active_noise_psd

        # A posteriori SNR: instantaneous SNR based on current observation
        gamma = np.maximum(Yp / curr_noise, eps)

        recursive = (prev_gain ** 2) * prev_Y_power / curr_noise
        ksi = alpha * recursive + (1.0 - alpha) * np.maximum(gamma - 1.0, 0.0)
        ksi = np.maximum(ksi, ksi_min)

        # Compute v parameter for LSA gain formula
        v = np.clip((ksi * gamma) / (1.0 + ksi), 1e-12, v_max)

        # Log-MMSE Gain (LSA)
        G_lsa = (ksi / (1.0 + ksi)) * np.exp(0.5 * expn(1, v))
        G_lsa = np.nan_to_num(G_lsa, nan=gain_floor, posinf=1.0, neginf=gain_floor)

        #Speech Presence Probability (SPP) calculation
        Lambda = (1.0 / (1.0 + ksi)) * np.exp(v)
        p_speech = np.clip(1.0 / (1.0 + q_val / ((1.0 - q_val) * Lambda + eps)), 1e-3, 1.0)

        # Combined gain: interpolate between LSA gain and gain floor based on SPP
        G[:, t:t + 1] = np.clip((G_lsa ** p_speech) * (gain_floor ** (1.0 - p_speech)), gain_floor, 1.0)

        # Update noise PSD estimate (only for non-oracle/adaptive case)
        if not is_adaptive:
            p_noise = 1.0 - p_speech
            estimated_noise = p_noise * Yp + p_speech * active_noise_psd
            active_noise_psd = noise_mu * active_noise_psd + (1.0 - noise_mu) * estimated_noise
            active_noise_psd = np.maximum(active_noise_psd, eps)

        prev_gain = G[:, t:t + 1]
        prev_Y_power = Yp

    # 5. Signal reconstruction
    S_hat = Y * G
    enhanced = librosa.istft(S_hat, hop_length=hop_length, win_length=n_fft)
    return librosa.util.fix_length(enhanced, size=original_length)
