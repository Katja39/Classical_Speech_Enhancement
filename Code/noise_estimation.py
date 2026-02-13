import numpy as np
import librosa
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

class NoiseEstimator(ABC):
    @abstractmethod
    def estimate(self, power_spectrum: np.ndarray, **kwargs) -> np.ndarray:
        pass

class PercentileNoiseEstimator(NoiseEstimator):
    def __init__(self, percentile=20.0, min_frames=10, max_fraction=0.30,
                 floor_rel=0.02, adaptive_short=True, **kwargs):
        self.percentile = percentile
        self.min_frames = min_frames
        self.max_fraction = max_fraction
        self.floor_rel = floor_rel
        self.adaptive_short = adaptive_short

    def estimate(self, power: np.ndarray, **kwargs) -> np.ndarray:
        """
        Estimate a static noise PSD (n_bins, 1) via percentile statistics over quiet frames
        """
        n_frames = power.shape[1]
        eps = kwargs.get('eps', 1e-10)

        # For very short signals, adapt the chosen percentile and the minimum number of frames
        # This avoids overly aggressive "quiet frame" selection when n_frames is small
        if self.adaptive_short and n_frames < 30:
            min_frames = max(2, n_frames // 4)
            target_frames = max(3, int(n_frames * 0.15))
            percentile = min(50.0, 100.0 * target_frames / n_frames)
        else:
            percentile = self.percentile
            min_frames = self.min_frames

        # Convert percentile to a target count of frames, then clamp to sensible bounds
        frames_by_percent = int(np.ceil(n_frames * (percentile / 100.0)))
        k = max(min_frames, frames_by_percent)
        k = min(k, max(1, int(np.ceil(n_frames * self.max_fraction))))
        k = min(k, n_frames)

        # Compute per-frame log-energy as a robust "quietness" measure
        frame_energy = np.mean(np.log(np.maximum(power, eps)), axis=0)

        # Pick the k quietest frames (lowest energy)
        quiet_frames = np.argsort(frame_energy)[:k]

        # For each frequency bin, take the chosen percentile over the selected quiet frames
        noise_psd = np.percentile(power[:, quiet_frames], percentile, axis=1, keepdims=True)

        # Apply a floor relative to the median spectrum (prevents unrealistically small PSD)
        signal_median = np.median(power, axis=1, keepdims=True)
        noise_psd = np.maximum(noise_psd, self.floor_rel * signal_median)

        return np.maximum(noise_psd, eps)


class MinTrackingNoiseEstimator(NoiseEstimator):
    def __init__(self, window_size=50, smoothing_factor=None, **kwargs):
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor

    def estimate(self, power: np.ndarray, **kwargs) -> np.ndarray:
        """
        Estimate a time-varying noise PSD (n_bins, n_frames) by min-tracking.
        """

        n_bins, n_frames = power.shape
        eps = kwargs.get('eps', 1e-10)

        # Choose smoothing alpha if not provided:
        alpha = self.smoothing_factor
        if alpha is None:
            alpha = max(0.8, min(0.95, 1 - 5 / n_frames))

        # IIR smoothing across time
        smoothed = np.zeros_like(power)
        smoothed[:, 0] = power[:, 0]

        for t in range(1, n_frames):
            smoothed[:, t] = alpha * smoothed[:, t - 1] + (1 - alpha) * power[:, t]

        # Minimum filter over time (per frequency bin)
        window = self._get_window_size(n_frames)
        from scipy.ndimage import minimum_filter1d
        minima = minimum_filter1d(smoothed, size=window, axis=1, mode='nearest')

        # Use the tracked minima as the noise PSD (time-varying)
        noise_psd = minima

        # Floor relative to median spectrum (avoid collapsing to near-zero)
        signal_median = np.median(power, axis=1, keepdims=True)
        noise_psd = np.maximum(noise_psd, 0.01 * signal_median)
        return np.maximum(noise_psd, eps)

    def _get_window_size(self, n_frames: int) -> int:
       window = min(max(3, self.window_size), n_frames)
       return window if window % 2 == 1 else window + 1

    def _sliding_minimum(self, data: np.ndarray, window: int) -> np.ndarray:
       n_bins, total_frames = data.shape
       result_frames = total_frames - window + 1
       minima = np.empty((n_bins, result_frames))
       for i in range(result_frames):
           minima[:, i] = np.min(data[:, i:i + window], axis=1)
       return minima

class TrueNoiseEstimator(NoiseEstimator):
    """Ground Truth estimation (Oracle)"""

    def __init__(self, **kwargs):
        pass

    def estimate(self, power: np.ndarray, **kwargs) -> np.ndarray:

        clean_audio = kwargs.get('clean_audio')
        noisy_audio = kwargs.get('noisy_audio')
        n_fft = kwargs.get('n_fft', 1024)
        hop_length = kwargs.get('hop_length', 256)
        win_length = kwargs.get('win_length', n_fft)
        eps = kwargs.get('eps', 1e-12)

        if clean_audio is None or noisy_audio is None:
            raise ValueError("TrueNoiseEstimator requires clean_audio and noisy_audio")

        # Ensure both signals have the same length
        min_len = min(len(clean_audio), len(noisy_audio))
        clean = np.asarray(clean_audio[:min_len], dtype=np.float64)
        noisy = np.asarray(noisy_audio[:min_len], dtype=np.float64)

        # Time-domain noise signal
        noise = noisy - clean

        # STFT of the noise signal -> noise PSD
        stft_noise = librosa.stft(
            noise,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=kwargs.get("window", "hann"),
            center=kwargs.get("center", True),
            pad_mode=kwargs.get("pad_mode", "reflect")
        )

        noise_psd = np.abs(stft_noise) ** 2
        noise_psd = np.maximum(noise_psd, eps)

        # Match frames if needed
        if noise_psd.shape[1] > power.shape[1]:
            noise_psd = noise_psd[:, :power.shape[1]]
        elif noise_psd.shape[1] < power.shape[1]:
            noise_psd = np.pad(noise_psd, ((0, 0), (0, power.shape[1] - noise_psd.shape[1])), mode='edge')

        return noise_psd


def noise_estimation(
        y: np.ndarray,
        sr: int,
        method: str = 'percentile',
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
        window="hann",
        center=True,
        pad_mode="reflect",
        **kwargs
) -> np.ndarray:
    if estimator_params is None:
        estimator_params = {}

    # Merge constructor params with any extra keyword args
    full_params = {**estimator_params, **kwargs}

    # Force mono if multi-channel input is provided
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # STFT -> power spectrogram
    win_length = win_length or n_fft
    stft = librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=center, pad_mode=pad_mode
    )
    power = np.abs(stft) ** 2

    n_bins, n_frames = power.shape
    eps = full_params.get('eps', 1e-10)

    # Very short signals
    if n_frames < 5:
        return _simple_noise_estimate(power, eps)

    estimator = _create_estimator(method, full_params)

    noise_psd = estimator.estimate(
        power,
        noisy_audio=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        **kwargs
    )

    return noise_psd


def _create_estimator(method: str, params: Dict[str, Any]) -> NoiseEstimator:
    if method == 'percentile':
        return PercentileNoiseEstimator(**params)
    elif method == 'min_tracking':
        return MinTrackingNoiseEstimator(**params)
    elif method == 'true_noise':
        return TrueNoiseEstimator(**params)
    else:
        raise ValueError(f"Unbekannte Methode: {method}")


def _simple_noise_estimate(power: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n_frames = power.shape[1]
    if n_frames < 2:
        noise_psd = np.mean(power, axis=1, keepdims=True)
    else:
        noise_psd = np.percentile(power, 25, axis=1, keepdims=True)
    return np.maximum(noise_psd, eps)