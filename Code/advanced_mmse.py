import numpy as np
import librosa
from scipy.special import expn  # exponential integral E1(x) = expn(1, x)

from evaluation_metrics import evaluate_audio_quality
from speech_enhancement_comparison import optimize_parameters
from load_files import load_clean_noisy, default_out_dir

from noise_estimation import noise_estimation

from parameter_ranges import param_ranges_omlsa

def advanced_mmse(noisy_audio, sr, n_fft, hop_length, alpha, ksi_min,
                  q, noise_mu, gain_floor, noise_percentile, noise_method,
                  v_max=80.0):
    """
    Ausgangspunkt: MMSE (STSA)

    OM-LSA / Log-MMSE-style enhancement:
    - Statt Amplitude wird das logarithmische Spektrum optimiert (Log-MMSE)

    - Decision-Directed a-priori SNR
    - LSA gain uses exp(0.5 * E1(v))
    - Speech Presence Probability (SPP) controls gain and noise update
    - Adaptive noise PSD update using SPP (simple and effective)
    """

    """ If speech is present → aggressive noise suppression; no speech → more conservative """

    noisy_audio = np.asarray(noisy_audio, dtype=np.float64)
    original_length = len(noisy_audio)
    eps = 1e-10

    # STFT
    Y = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    Y_power = np.abs(Y) ** 2

    # Initial noise PSD
    noise_psd = noise_estimation(
        noisy_audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft,
        percentile=noise_percentile,
        method=noise_method,
        eps=eps
    )

    noise_psd = np.maximum(noise_psd, eps)

    num_bins, num_frames = Y.shape
    G = np.zeros((num_bins, num_frames), dtype=np.float64)

    # Prior speech presence probability
    q_val = float(np.clip(q, 1e-3, 1 - 1e-3))

    Yp_0 = Y_power[:, 0:1]
    gamma_0 = Yp_0 / noise_psd
    gamma_0 = np.maximum(gamma_0, eps)
    ksi_0 = np.maximum(gamma_0 - 1.0, ksi_min)

    v_0 = (ksi_0 * gamma_0) / (1.0 + ksi_0)
    v_0 = np.clip(v_0, 1e-12, v_max)

    # Log-MMSE gain
    E1_0 = expn(1, v_0)
    G_lsa_0 = (ksi_0 / (1.0 + ksi_0)) * np.exp(0.5 * E1_0)
    G_lsa_0 = np.nan_to_num(G_lsa_0, nan=gain_floor, posinf=1.0, neginf=gain_floor)

    # Speech presence probability
    Lambda_0 = (1.0 / (1.0 + ksi_0)) * np.exp(v_0)
    p_speech_0 = 1.0 / (1.0 + ((1.0 - q_val) / q_val) * Lambda_0)
    p_speech_0 = np.clip(p_speech_0, 1e-3, 1.0)
    p_noise_0 = 1.0 - p_speech_0

    # OM-LSA gain
    G_0 = (G_lsa_0 ** p_speech_0) * (gain_floor ** (1.0 - p_speech_0))
    G_0 = np.clip(G_0, gain_floor, 1.0)

    G[:, 0:1] = G_0

    # Noise update für t=0
    estimated_noise_0 = p_noise_0 * Yp_0 + p_speech_0 * noise_psd
    noise_psd = noise_mu * noise_psd + (1.0 - noise_mu) * estimated_noise_0
    noise_psd = np.maximum(noise_psd, eps)

    prev_gain = G_0
    prev_Y_power = Yp_0

    for t in range(1, num_frames):
        Yp = Y_power[:, t:t + 1]
        gamma = Yp / noise_psd
        gamma = np.maximum(gamma, eps)

        # Decision-directed a-priori SNR
        recursive = (prev_gain ** 2) * prev_Y_power / noise_psd
        direct = np.maximum(gamma - 1.0, 0.0)
        ksi = alpha * recursive + (1.0 - alpha) * direct
        ksi = np.maximum(ksi, ksi_min)

        v = (ksi * gamma) / (1.0 + ksi)
        v = np.clip(v, 1e-12, v_max)

        # Log-MMSE gain
        E1 = expn(1, v)
        G_lsa = (ksi / (1.0 + ksi)) * np.exp(0.5 * E1)
        G_lsa = np.nan_to_num(G_lsa, nan=gain_floor, posinf=1.0, neginf=gain_floor)

        # Speech presence probability
        Lambda = (1.0 / (1.0 + ksi)) * np.exp(v)
        p_speech = 1.0 / (1.0 + ((1.0 - q_val) / q_val) * Lambda)
        p_speech = np.clip(p_speech, 1e-3, 1.0)
        p_noise = 1.0 - p_speech

        # OM-LSA gain
        G_t = (G_lsa ** p_speech) * (gain_floor ** (1.0 - p_speech))
        G_t = np.clip(G_t, gain_floor, 1.0)

        G[:, t:t + 1] = G_t

        # Adaptive noise PSD update (Cohen & Berdugo 2001)
        estimated_noise = p_noise * Yp + p_speech * noise_psd
        noise_psd = noise_mu * noise_psd + (1.0 - noise_mu) * estimated_noise
        noise_psd = np.maximum(noise_psd, eps)

        prev_gain = G_t
        prev_Y_power = Yp

    # ISTFT
    S_hat = Y * G
    clean_audio = librosa.istft(S_hat, hop_length=hop_length, win_length=n_fft)

    if len(clean_audio) > original_length:
        clean_audio = clean_audio[:original_length]
    elif len(clean_audio) < original_length:
        clean_audio = np.pad(clean_audio, (0, original_length - len(clean_audio)))

    return clean_audio
