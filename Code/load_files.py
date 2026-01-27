import os
import librosa

def load_clean_noisy(clean_path, noisy_path, target_sr=16000):
    clean, sr_c = librosa.load(clean_path, sr=None)
    noisy, sr_n = librosa.load(noisy_path, sr=None)

    if sr_c != target_sr:
        clean = librosa.resample(clean, orig_sr=sr_c, target_sr=target_sr)
    if sr_n != target_sr:
        noisy = librosa.resample(noisy, orig_sr=sr_n, target_sr=target_sr)

    L = min(len(clean), len(noisy))
    clean = clean[:L]
    noisy = noisy[:L]
    return clean, noisy, target_sr


def default_out_dir(base_dir, folder_name):
    out_dir = os.path.join(base_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
