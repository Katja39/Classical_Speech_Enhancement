import os
import numpy as np
import librosa
import soundfile as sf
from scipy.special import expn  # exponential integral E1(x) = expn(1, x)

from evaluation_metrics import evaluate_audio_quality, optimize_parameters
from load_files import load_clean_noisy, default_out_dir

def advanced_mmse(noisy_audio, sr,
                      noise_start=0.0, noise_end=0.1,
                      n_fft=1024, hop_length=256,
                      alpha=0.98, ksi_min=0.001,
                      q=0.5,                 # prior speech presence prob
                      noise_mu=0.98,         # noise PSD smoothing (close to 1.0 = slow update)
                      gain_floor=0.05,       # minimum gain (prevents musical noise bursts)
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

    # Noise segment for initialization
    ns = int(max(0.0, noise_start) * sr)
    ne = int(max(0.0, noise_end) * sr)
    ne = min(ne, original_length)
    if ne <= ns:
        ns, ne = 0, min(int(0.1 * sr), original_length)

    noise_segment = noisy_audio[ns:ne]
    if len(noise_segment) < 2:
        noise_segment = noisy_audio[:min(original_length, max(2, int(0.1 * sr)))]

    # STFT
    stft_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    Y = stft_noisy
    Y_power = np.abs(Y) ** 2

    stft_noise = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    noise_psd = np.mean(np.abs(stft_noise) ** 2, axis=1, keepdims=True)
    noise_psd = np.maximum(noise_psd, eps)

    num_bins, num_frames = Y.shape

    G = np.zeros((num_bins, num_frames), dtype=np.float64)

    prev_gain = None
    prev_Y_power = None

    # Clamp q
    q = float(np.clip(q, 1e-3, 1 - 1e-3))

    for t in range(num_frames):
        Yp = Y_power[:, t:t+1]

        # a-posteriori SNR gamma
        gamma = Yp / noise_psd
        gamma = np.maximum(gamma, eps)

        # a-priori SNR ksi (Decision-Directed)
        if t == 0:
            ksi = np.maximum(gamma - 1.0, ksi_min)
        else:
            recursive = (prev_gain ** 2) * prev_Y_power / noise_psd
            direct = np.maximum(gamma - 1.0, 0.0)
            ksi = alpha * recursive + (1.0 - alpha) * direct
            ksi = np.maximum(ksi, ksi_min)

        # v = ksi*gamma/(1+ksi) used in LSA and SPP
        v = (ksi * gamma) / (1.0 + ksi)
        v = np.clip(v, 1e-12, v_max)

        # ---- LSA (Log-MMSE) gain ----
        # G_lsa = (ksi/(1+ksi)) * exp(0.5 * E1(v))
        E1 = expn(1, v)  # E1(v)
        G_lsa = (ksi / (1.0 + ksi)) * np.exp(0.5 * E1)
        G_lsa = np.nan_to_num(G_lsa, nan=gain_floor, posinf=1.0, neginf=gain_floor)

        # ---- Speech Presence Probability (SPP) ----
        # Likelihood ratio (common form):
        # Lambda = (1/(1+ksi)) * exp(v)
        Lambda = (1.0 / (1.0 + ksi)) * np.exp(v)
        # p = 1 / (1 + ((1-q)/q) * Lambda)
        p_speech = 1.0 / (1.0 + ((1.0 - q) / q) * Lambda)
        p_speech = np.clip(p_speech, 1e-3, 1.0)

        # OM-LSA style combination:
        # Use geometric mixing to keep it smooth:
        # G = G_lsa^p * G_floor^(1-p)
        G_t = (G_lsa ** p_speech) * (gain_floor ** (1.0 - p_speech))
        G_t = np.clip(G_t, gain_floor, 1.0)

        G[:, t:t+1] = G_t

        # ---- Adaptive noise PSD update using SPP ----
        # Update towards current power when speech absence prob is high.
        p_noise = 1.0 - p_speech
        noise_psd = (noise_mu + (1.0 - noise_mu) * p_speech) * noise_psd + (1.0 - noise_mu) * p_noise * Yp
        noise_psd = np.maximum(noise_psd, eps)

        prev_gain = G_t
        prev_Y_power = Yp

    # Apply gain (keep noisy phase)
    S_hat = Y * G
    clean_audio = librosa.istft(S_hat, hop_length=hop_length, win_length=n_fft)

    # Length fix
    if len(clean_audio) > original_length:
        clean_audio = clean_audio[:original_length]
    elif len(clean_audio) < original_length:
        clean_audio = np.pad(clean_audio, (0, original_length - len(clean_audio)))

    return clean_audio


def test_advanced_mmse(clean_path=None, noisy_path=None, data_dir="data", out_dir=None, stem="sample"):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if clean_path is None or noisy_path is None:
        data_dir = os.path.join(base_dir, data_dir)
        clean_path = os.path.join(data_dir, "p232_014_clean.wav")
        noisy_path = os.path.join(data_dir, "p232_014_noiseWithMusic.wav")
        stem = "p232_014"

    if out_dir is None:
        out_dir = default_out_dir(base_dir, "results_omlsa")

    clean_reference, noisy, sr = load_clean_noisy(clean_path, noisy_path, target_sr=16000)

    param_ranges = {
        "alpha": [0.90, 0.94, 0.96, 0.98],#0.92
        "n_fft": [512, 1024],
        "hop_length": [128, 256],
        "ksi_min": [0.001, 0.01, 0.05],
        "gain_floor": [0.02, 0.05, 0.08, 0.1],
        "noise_mu": [0.95, 0.98, 0.99], #0.97
        "q": [0.3, 0.5, 0.7],
    }

    print("\nStarting OM-LSA/Log-MMSE parameter optimization...")
    optimization_results = optimize_parameters(clean_reference, noisy, sr, advanced_mmse, param_ranges)

    stoi_results = evaluate_audio_quality(clean_reference, noisy, optimization_results['stoi']['enhanced'], sr,
                                         "OM-LSA (STOI optimized)")
    pesq_results = evaluate_audio_quality(clean_reference, noisy, optimization_results['pesq']['enhanced'], sr,
                                         "OM-LSA (PESQ optimized)")

    stoi_path = os.path.join(out_dir, f"{stem}_omlsa_optimized_stoi.wav")
    pesq_path = os.path.join(out_dir, f"{stem}_omlsa_optimized_pesq.wav")
    sf.write(stoi_path, optimization_results['stoi']['enhanced'], sr)
    sf.write(pesq_path, optimization_results['pesq']['enhanced'], sr)

    return noisy, optimization_results, sr, stoi_results


def main():
    print("OM-LSA / Log-MMSE Speech Enhancement")
    print("=" * 60)
    try:
        noisy, optimization_results, sr, results = test_advanced_mmse()

        if results and optimization_results:
            print("\n" + "=" * 40)
            print("FINAL SUMMARY - OM-LSA")
            print("=" * 40)
            if "stoi_noisy" in results:
                print(f"STOI: {results['stoi_noisy']:.4f} -> {results['stoi_enhanced']:.4f} (+{results['stoi_improvement']:.4f})")
            print(f"PESQ STOI-optimized: {optimization_results['stoi']['score']:.2f}")
            print(f"PESQ PESQ-optimized: {optimization_results['pesq']['score']:.2f}")
            print("=" * 40)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
