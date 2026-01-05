import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spectralSubtractor import test_spectral_subtraction
from mmse import test_mmse

def compare_algorithms():
    """
    Compare Spectral Subtraction and MMSE algorithms
    """
    print("=" * 70)
    print("COMPARISON OF SPEECH ENHANCEMENT ALGORITHMS")
    print("=" * 70)

    # Test Spectral Subtraction
    print("\n1. SPECTRAL SUBTRACTION")
    print("-" * 40)
    ss_noisy, ss_results, ss_sr, ss_eval = test_spectral_subtraction()

    # Test MMSE
    print("\n\n2. MMSE ESTIMATOR")
    print("-" * 40)
    mmse_noisy, mmse_results, mmse_sr, mmse_eval = test_mmse()

    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print("\nSTOI COMPARISON (higher is better):")
    if ss_eval and 'stoi_enhanced' in ss_eval:
        print(f"  Spectral Subtraction: {ss_eval['stoi_enhanced']:.4f}")
    if mmse_eval and 'stoi_enhanced' in mmse_eval:
        print(f"  MMSE:       {mmse_eval['stoi_enhanced']:.4f}")

    print("\nPESQ COMPARISON (higher is better):")
    if ss_results and 'stoi' in ss_results and 'score' in ss_results['stoi']:
        print(f"  Spectral Subtraction (STOI-opt): {ss_results['stoi']['score']:.2f}")
    if ss_results and 'pesq' in ss_results and 'score' in ss_results['pesq']:
        print(f"  Spectral Subtraction (PESQ-opt): {ss_results['pesq']['score']:.2f}")
    if mmse_results and 'stoi' in mmse_results and 'score' in mmse_results['stoi']:
        print(f"  MMSE (STOI-opt):      {mmse_results['stoi']['score']:.2f}")
    if mmse_results and 'pesq' in mmse_results and 'score' in mmse_results['pesq']:
        print(f"  MMSE (PESQ-opt):      {mmse_results['pesq']['score']:.2f}")

    print("\n" + "=" * 70)


def main():
    compare_algorithms()


if __name__ == "__main__":
    main()