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
        n_frames = power.shape[1]
        eps = kwargs.get('eps', 1e-10)

        if self.adaptive_short and n_frames < 30:
            min_frames = max(2, n_frames // 4)
            target_frames = max(3, int(n_frames * 0.15))
            percentile = min(50.0, 100.0 * target_frames / n_frames)
        else:
            percentile = self.percentile
            min_frames = self.min_frames

        frames_by_percent = int(np.ceil(n_frames * (percentile / 100.0)))
        k = max(min_frames, frames_by_percent)
        k = min(k, max(1, int(np.ceil(n_frames * self.max_fraction))))
        k = min(k, n_frames)

        frame_energy = np.mean(np.log(np.maximum(power, eps)), axis=0)
        quiet_frames = np.argsort(frame_energy)[:k]

        noise_psd = np.median(power[:, quiet_frames], axis=1, keepdims=True)

        signal_median = np.median(power, axis=1, keepdims=True)
        noise_psd = np.maximum(noise_psd, self.floor_rel * signal_median)

        return np.maximum(noise_psd, eps)


class MinTrackingNoiseEstimator(NoiseEstimator):
    def __init__(self, window_size=50, smoothing_factor=None, **kwargs):
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor

    def estimate(self, power: np.ndarray, **kwargs) -> np.ndarray:
        n_bins, n_frames = power.shape
        eps = kwargs.get('eps', 1e-10)

        if self.smoothing_factor is None:
            self.smoothing_factor = max(0.8, min(0.95, 1 - 5 / n_frames))

        smoothed = np.zeros_like(power)
        smoothed[:, 0] = power[:, 0]
        alpha = self.smoothing_factor

        for t in range(1, n_frames):
            smoothed[:, t] = alpha * smoothed[:, t - 1] + (1 - alpha) * power[:, t]

        window = self._get_window_size(n_frames)
        pad = window // 2
        padded = np.pad(smoothed, ((0, 0), (pad, pad)), mode='edge')

        minima = self._sliding_minimum(padded, window)

        noise_psd = np.median(minima, axis=1, keepdims=True)

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

        if clean_audio is None or noisy_audio is None:
            raise ValueError("TrueNoiseEstimator requires clean_audio and noisy_audio")

        min_len = min(len(clean_audio), len(noisy_audio))
        clean = clean_audio[:min_len]
        noisy = noisy_audio[:min_len]

        stft_clean = librosa.stft(clean, n_fft=n_fft, hop_length=hop_length)
        stft_noisy = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)

        power_clean = np.abs(stft_clean) ** 2
        power_noisy = np.abs(stft_noisy) ** 2

        noise_psd = np.maximum(power_noisy - power_clean, 1e-10)

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
        **kwargs
) -> np.ndarray:
    if estimator_params is None:
        estimator_params = {}

    full_params = {**estimator_params, **kwargs}

    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    win_length = win_length or n_fft
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    power = np.abs(stft) ** 2

    n_bins, n_frames = power.shape
    eps = full_params.get('eps', 1e-10)

    if n_frames < 5:
        return _simple_noise_estimate(power, eps)

    estimator = _create_estimator(method, full_params)

    noise_psd = estimator.estimate(
        power,
        noisy_audio=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
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