import numpy as np
import librosa

def noise_estimation(y, sr, n_fft=1024, hop_length=256, win_length=None,
                     percentile=20.0, min_frames=10, max_fraction=0.30,
                     floor_rel=0.02, smooth_bins=7, adaptive_short=True,
                     eps=1e-10, debug=False, **kwargs):
    """
    Universelle Rauschschätzung mit mehreren Methoden

    Extra Parameter für erweiterte Methoden:
    - method: 'percentile' (default), 'min_tracking', 'voice_activity'
    - vad_threshold: Für voice_activity Methode
    - window_size: Für min_tracking Methode
    """

    # Extrahiere optionale Methoden-Parameter
    method = kwargs.get('method', 'percentile')
    vad_threshold = kwargs.get('vad_threshold', percentile)


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

    else:  # Default: percentile method (kompatibel)
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
    """Minimum statistics method"""
    window_size = min(window_size, n_frames // 2)

    noise_psd = np.zeros((power.shape[0], 1))
    for k in range(power.shape[0]):
        min_values = []
        for t in range(n_frames):
            start = max(0, t - window_size // 2)
            end = min(n_frames, t + window_size // 2)
            min_val = np.min(power[k, start:end])
            min_values.append(min_val)

        noise_psd[k] = np.median(min_values)

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