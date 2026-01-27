param_ranges_ss = {
    "alpha": [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], #3 - 6
    "beta": [0.001, 0.005, 0.05, 0.1, 0.15], #SNR = -5 0.02 to 0.06; SNR = +5, 0.005 to 0.02 (Berouti)
    "n_fft": [512, 1024],
    "hop_length": [128, 256],
    "noise_percentile": [10.0, 20.0, 30.0],
     "noise_method": ["true_noise", "percentile", "min_tracking"]
}

param_ranges_mmse = {
    "alpha": [0.85, 0.90, 0.95, 0.99],#0.98 #Decision- Directed smoothing factor
    "ksi_min": [0.001, 0.01, 0.02, 0.05, 0.1, 0.2],
    "gain_min": [0.01, 0.02, 0.05, 0.1, 0.2],
    "gain_max": [1.0],
    "n_fft": [512, 1024],
    "hop_length": [128, 256],
    "noise_percentile": [10.0, 20.0, 30.0],
    "noise_method": ["true_noise", "percentile", "min_tracking"]
}

param_ranges_wiener = {
    "alpha": [0.85, 0.90, 0.95, 0.98],
    "gain_floor": [0.02, 0.05, 0.1, 0.2],
    "n_fft": [512, 1024],
    "hop_length": [128, 256],
    "noise_percentile": [10.0, 20.0, 30.0],
     "noise_method": ["true_noise", "percentile", "min_tracking"]
}

param_ranges_omlsa = {
    "alpha": [0.6, 0.7, 0.80, 0.9, 0.95],
    "ksi_min": [0.001, 0.005, 0.01, 0.05],
    "gain_floor": [0.02, 0.05, 0.1, 0.2],
    "noise_mu": [0.92, 0.95, 0.98],
    "q": [0.2, 0.4, 0.6, 0.8],
    "n_fft": [512, 1024],
    "hop_length": [128, 256],
    "noise_percentile": [10.0, 20.0, 30.0],
     "noise_method": ["true_noise", "percentile", "min_tracking"]
}