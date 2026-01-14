import numpy as np
import librosa

def noise_estimation(y, sr, n_fft=1024, hop_length=256, win_length=None,
                     percentile=20.0, min_frames=10, max_fraction=0.30,
                     floor_rel=0.02, smooth_bins=7, adaptive_short=True,
                     eps=1e-10, debug=False, **kwargs):

    method = kwargs.get('method', 'percentile')
    vad_threshold = kwargs.get('vad_threshold', percentile)

    # Check for true_noise method
    if method == 'true_noise':
        if debug:
            print("Using TRUE NOISE method (ground truth)")

        clean_audio = kwargs.get('clean_audio', None)
        if clean_audio is None:
            raise ValueError("'true_noise' method requires 'clean_audio' parameter")

        return calculate_true_noise_psd(
            clean_audio, y, sr, n_fft=n_fft, hop_length=hop_length
        )

    y = np.asarray(y, dtype=np.float64)

    if win_length is None:
        win_length = n_fft

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # STFT
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    power = np.abs(stft) ** 2
    n_bins, n_frames = power.shape

    window_size = kwargs.get('window_size', min(50, int(n_frames * 0.25) if 'n_frames' in locals() else 50))

    # Kurze Signale
    if n_frames < 2:
        noise_psd = np.mean(power, axis=1, keepdims=True)
        return np.maximum(noise_psd, eps)

    if n_frames < 5:
        noise_psd = np.percentile(power, 25, axis=1, keepdims=True)
        return np.maximum(noise_psd, eps)

    # Methoden-Dispatch
    if method == 'min_tracking' and n_frames > 10:
        return _noise_min_tracking(power, n_frames, window_size, eps)

    elif method == 'voice_activity' and n_frames > 5:
        return _noise_voice_activity(power, n_frames, vad_threshold, min_frames, eps)

    else:  # Default: percentile method
        return _noise_percentile(power, n_frames, percentile, min_frames,
                                 max_fraction, floor_rel, adaptive_short, eps)


def _noise_percentile(power, n_frames, percentile=20.0, min_frames=10,
                      max_fraction=0.30, floor_rel=0.02, adaptive_short=True, eps=1e-10):
    """Original percentile method for backwards compatibility"""
    if adaptive_short and n_frames < 30:
        min_frames = max(2, n_frames // 4)
        target_frames = max(3, int(n_frames * 0.15))
        percentile = min(50.0, 100.0 * target_frames / n_frames)

    frames_by_percent = int(np.ceil(n_frames * (percentile / 100.0)))
    k = max(min_frames, frames_by_percent)

    k_max = max(1, int(np.ceil(n_frames * max_fraction)))
    k = min(k, k_max)
    k = min(k, n_frames)

    frame_energy = np.mean(np.log(power + eps), axis=0)
    idx = np.argsort(frame_energy)[:k]

    selected = power[:, idx]
    noise_psd = np.median(selected, axis=1, keepdims=True)

    per_bin_med = np.median(power, axis=1, keepdims=True)
    noise_psd = np.maximum(noise_psd, floor_rel * per_bin_med)
    noise_psd = np.maximum(noise_psd, eps)

    return noise_psd

def _noise_min_tracking(power, n_frames, window_size=50, eps=1e-10):
    """
    - Temporal IIR smoothing per frequency bin (reduces random dips)
    - Sliding-window minimum (running minimum), vectorized
    - Robust aggregation (median over time)
    """
    power = np.asarray(power, dtype=np.float64)
    n_bins, T = power.shape
    if T != n_frames:
        n_frames = T

    # Handle very short signals robustly
    if n_frames < 2:
        noise_psd = np.mean(power, axis=1, keepdims=True)
        return np.maximum(noise_psd, eps)

    # Clamp window size to valid range and make it odd
    window_size = int(window_size)
    window_size = max(3, min(window_size, n_frames))
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2

    # 1) Temporal smoothing (IIR / exponential moving average)
    # Fixed smoothing factor to keep your parameter search unchanged
    smooth = 0.90  # Higher -> smoother (more robust), lower -> more adaptive
    p_smooth = np.empty_like(power)
    p_smooth[:, 0] = power[:, 0]
    one_minus = 1.0 - smooth
    for t in range(1, n_frames):
        p_smooth[:, t] = smooth * p_smooth[:, t - 1] + one_minus * power[:, t]

    # 2) Running minimum within a sliding window (vectorized)
    padded = np.pad(p_smooth, ((0, 0), (pad, pad)), mode="edge")
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, window_shape=window_size, axis=1)  # (bins, frames, win)
        pmin_t = np.min(windows, axis=2)  # (bins, frames)
    except Exception:
        # Fallback: compute per-frame minima (still avoids looping over bins)
        pmin_t = np.empty((n_bins, n_frames), dtype=np.float64)
        for t in range(n_frames):
            pmin_t[:, t] = np.min(padded[:, t:t + window_size], axis=1)

    # 3) Convert time-varying minima to a single stationary noise PSD (median over time)
    noise_psd = np.median(pmin_t, axis=1, keepdims=True)

    # 4) Safety floor to avoid severe underestimation (same spirit as your original code)
    signal_median = np.median(power, axis=1, keepdims=True)
    noise_psd = np.maximum(noise_psd, 0.01 * signal_median)
    noise_psd = np.maximum(noise_psd, eps)

    return noise_psd

def _noise_voice_activity(power, n_frames, threshold_percentile=30,
                          min_frames=5, eps=1e-10):
    """Voice activity detection method"""
    frame_energy = np.mean(power, axis=0)
    energy_threshold = np.percentile(frame_energy, threshold_percentile)

    noise_frames = frame_energy <= energy_threshold

    if np.sum(noise_frames) < min_frames:
        sorted_indices = np.argsort(frame_energy)
        noise_frames_indices = sorted_indices[:max(min_frames, n_frames // 10)]
        noise_power = power[:, noise_frames_indices]
    else:
        noise_power = power[:, noise_frames]

    noise_psd = np.median(noise_power, axis=1, keepdims=True)

    signal_median = np.median(power, axis=1, keepdims=True)
    noise_psd = np.maximum(noise_psd, 0.01 * signal_median)
    noise_psd = np.maximum(noise_psd, eps)

    return noise_psd


def calculate_true_noise_psd(clean_audio, noisy_audio, sr, n_fft=1024, hop_length=256):
    """
    Calculate noise PSD from clean and noisy audio
    """
    # Ensure same length
    min_len = min(len(clean_audio), len(noisy_audio))
    clean_audio = clean_audio[:min_len]
    noisy_audio = noisy_audio[:min_len]

    noise_time = noisy_audio - clean_audio

    # STFT of clean and noisy
    stft_clean = librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    stft_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)

    # True noise in STFT domain
    stft_noise = stft_noisy - stft_clean

    # Noise Power Spectral Density
    noise_psd = np.abs(stft_noise) ** 2

    # Average over frames for initial estimate
    noise_psd_mean = np.mean(noise_psd, axis=1, keepdims=True)

    return noise_psd_mean

