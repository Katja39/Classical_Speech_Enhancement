import os
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spectralSubtractor import test_spectral_subtraction
from mmse import test_mmse
from wiener_filter import test_wiener
from advanced_mmse import test_advanced_mmse

#Comparison for a single file

def find_pairs(data_dir: str):
    wavs = [f for f in os.listdir(data_dir) if f.lower().endswith(".wav")]
    clean_files = [f for f in wavs if "_clean" in f.lower()]

    pairs = []
    for cf in clean_files:
        stem = re.sub(r"(?i)_clean\.wav$", "", cf)
        candidates = [
            f"{stem}_noisy.wav",
            f"{stem}_noise.wav",
            f"{stem}_noiseWithMusic.wav",
            f"{stem}_noisewithmusic.wav",
        ]
        fallback = [f for f in wavs
                    if f.lower().startswith(stem.lower())
                    and ("noise" in f.lower() or "noisy" in f.lower())
                    and f.lower() != cf.lower()]

        noisy = None
        for c in candidates:
            if c in wavs:
                noisy = c
                break
        if noisy is None and len(fallback) == 1:
            noisy = fallback[0]

        if noisy is not None:
            pairs.append({
                "stem": stem,
                "clean": os.path.join(data_dir, cf),
                "noisy": os.path.join(data_dir, noisy)
            })

    return pairs


def compare_algorithms():
    """
    Compare algorithms on ONE pair from ./data
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    pairs = find_pairs(data_dir)
    if not pairs:
        raise RuntimeError("No pairs found in ./data (expected *_clean.wav + *_noise*.wav).")

    STEM_TO_TEST = None  # z.B. "p232_014"

    if STEM_TO_TEST is not None:
        candidates = [p for p in pairs if p["stem"] == STEM_TO_TEST]
        if not candidates:
            raise RuntimeError(f"STEM '{STEM_TO_TEST}' not found in ./data.")
        pair = candidates[0]
    else:
        pair = pairs[0]  # erstes Pair

    stem = pair["stem"]
    clean_path = pair["clean"]
    noisy_path = pair["noisy"]

    print("=" * 70)
    print("COMPARISON OF SPEECH ENHANCEMENT ALGORITHMS (SINGLE FILE)")
    print("=" * 70)
    print(f"Stem : {stem}")
    print(f"Clean: {os.path.basename(clean_path)}")
    print(f"Noisy: {os.path.basename(noisy_path)}")

    # 1) Spectral Subtraction
    print("\n1. SPECTRAL SUBTRACTION")
    print("-" * 40)
    ss_noisy, ss_results, ss_sr, ss_eval = test_spectral_subtraction(
        clean_path=clean_path, noisy_path=noisy_path, stem=stem
    )

    # 2) MMSE
    print("\n\n2. MMSE ESTIMATOR")
    print("-" * 40)
    mmse_noisy, mmse_results, mmse_sr, mmse_eval = test_mmse(
        clean_path=clean_path, noisy_path=noisy_path, stem=stem
    )

    # 3) Wiener Filter
    print("\n\n3. WIENER FILTER")
    print("-" * 40)
    wiener_noisy, wiener_results, wiener_sr, wiener_eval = test_wiener(
        clean_path=clean_path, noisy_path=noisy_path, stem=stem
    )

    # 4) OM-LSA / Log-MMSE
    print("\n\n4. LOG-MMSE / OM-LSA")
    print("-" * 40)
    omlsa_noisy, omlsa_results, omlsa_sr, omlsa_eval = test_advanced_mmse(
        clean_path=clean_path, noisy_path=noisy_path, stem=stem
    )

    # ================= SUMMARY =================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    def safe_get(d, key):
        return d.get(key) if isinstance(d, dict) else None

    print("\nSTOI COMPARISON (higher is better):")
    if safe_get(ss_eval, "stoi_enhanced") is not None:
        print(f"  Spectral Subtraction: {ss_eval['stoi_enhanced']:.4f}")
    if safe_get(mmse_eval, "stoi_enhanced") is not None:
        print(f"  MMSE:                 {mmse_eval['stoi_enhanced']:.4f}")
    if safe_get(wiener_eval, "stoi_enhanced") is not None:
        print(f"  Wiener Filter:        {wiener_eval['stoi_enhanced']:.4f}")
    if safe_get(omlsa_eval, "stoi_enhanced") is not None:
        print(f"  OM-LSA / Log-MMSE:    {omlsa_eval['stoi_enhanced']:.4f}")

    print("\nPESQ COMPARISON (higher is better):")
    if ss_results and "stoi" in ss_results:
        print(f"  Spectral Subtraction (STOI-opt): {ss_results['stoi']['score']:.2f}")
        print(f"  Spectral Subtraction (PESQ-opt): {ss_results['pesq']['score']:.2f}")
    if mmse_results and "stoi" in mmse_results:
        print(f"  MMSE (STOI-opt):                 {mmse_results['stoi']['score']:.2f}")
        print(f"  MMSE (PESQ-opt):                 {mmse_results['pesq']['score']:.2f}")
    if wiener_results and "stoi" in wiener_results:
        print(f"  Wiener Filter (STOI-opt):        {wiener_results['stoi']['score']:.2f}")
        print(f"  Wiener Filter (PESQ-opt):        {wiener_results['pesq']['score']:.2f}")
    if omlsa_results and "stoi" in omlsa_results:
        print(f"  OM-LSA / Log-MMSE (STOI-opt):    {omlsa_results['stoi']['score']:.2f}")
        print(f"  OM-LSA / Log-MMSE (PESQ-opt):    {omlsa_results['pesq']['score']:.2f}")

    print("\n" + "=" * 70)


def main():
    compare_algorithms()


if __name__ == "__main__":
    main()
