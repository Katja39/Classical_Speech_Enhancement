import numpy as np
import librosa
from scipy.special import expn  # exponential integral E1(x) = expn(1, x)

from noise_estimation import noise_estimation

def advanced_mmse(noisy_audio, sr, n_fft, hop_length, alpha, ksi_min,
                  q, noise_mu, gain_floor, noise_percentile, noise_method, clean_audio=None,
                  v_max=80.0):
    """
    Ausgangspunkt: MMSE (STSA)

    OM-LSA / Log-MMSE-style enhancement:
    - Statt Amplitude wird das logarithmische Spektrum optimiert (Log-MMSE)

    - Decision-Directed a-priori SNR
    - LSA gain uses exp(0.5 * E1(v))
    - Speech Presence Probability (SPP) controls gain and noise update
    - Adaptive noise PSD update using SPP
    """

    """ If speech is present → aggressive noise suppression; no speech → more conservative """

    noisy_audio = np.asarray(noisy_audio, dtype=np.float64)
    original_length = len(noisy_audio)
    eps = 1e-10

    # 1. Zeit-Frequenz-Analyse
    Y = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    Y_power = np.abs(Y) ** 2
    num_bins, num_frames = Y.shape

    # 2. Rauschschätzung (Vektor oder Matrix bei TrueNoise)
    noise_psd_all = noise_estimation(
        noisy_audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, percentile=noise_percentile,
        method=noise_method, clean_audio=clean_audio, eps=eps
    )

    is_adaptive = (noise_psd_all.shape[1] > 1)
    G = np.zeros((num_bins, num_frames), dtype=np.float64)
    q_val = float(np.clip(q, 1e-3, 1 - 1e-3))

    # Initialisierung
    active_noise_psd = noise_psd_all[:, 0:1] if is_adaptive else noise_psd_all
    prev_gain = np.ones((num_bins, 1)) * gain_floor
    prev_Y_power = Y_power[:, 0:1]

    # 3. Haupt-Loop über alle Frames
    for t in range(num_frames):
        Yp = Y_power[:, t:t + 1]
        curr_noise = noise_psd_all[:, t:t + 1] if is_adaptive else active_noise_psd

        # A-posteriori SNR
        gamma = np.maximum(Yp / curr_noise, eps)

        # Decision-Directed A-priori SNR (ksi)
        recursive = (prev_gain ** 2) * prev_Y_power / curr_noise
        ksi = alpha * recursive + (1.0 - alpha) * np.maximum(gamma - 1.0, 0.0)
        ksi = np.maximum(ksi, ksi_min)

        # Hilfsvariable v für das Exponential-Integral
        v = np.clip((ksi * gamma) / (1.0 + ksi), 1e-12, v_max)

        # Log-MMSE Gain (LSA)
        G_lsa = (ksi / (1.0 + ksi)) * np.exp(0.5 * expn(1, v))
        G_lsa = np.nan_to_num(G_lsa, nan=gain_floor, posinf=1.0, neginf=gain_floor)

        # Sprachpräsenzwahrscheinlichkeit (SPP)
        Lambda = (1.0 / (1.0 + ksi)) * np.exp(v)
        p_speech = np.clip(1.0 / (1.0 + q_val / ((1.0 - q_val) * Lambda + eps)), 1e-3, 1.0)

        # Kombinierter Gain
        G[:, t:t + 1] = np.clip((G_lsa ** p_speech) * (gain_floor ** (1.0 - p_speech)), gain_floor, 1.0)

        # Rausch-Update (nur wenn kein Oracle/TrueNoise)
        if not is_adaptive:
            p_noise = 1.0 - p_speech
            estimated_noise = p_noise * Yp + p_speech * active_noise_psd
            active_noise_psd = noise_mu * active_noise_psd + (1.0 - noise_mu) * estimated_noise
            active_noise_psd = np.maximum(active_noise_psd, eps)

        prev_gain = G[:, t:t + 1]
        prev_Y_power = Yp

    # 4. Rekonstruktion
    S_hat = Y * G
    enhanced = librosa.istft(S_hat, hop_length=hop_length, win_length=n_fft)
    return librosa.util.fix_length(enhanced, size=original_length)
