import numpy as np


def alpha_min_from_snr(C: float, snr: float):
    """
    Paper-style scaling: alpha_min ~ 1/sqrt(C * SNR).
    """
    return 1.0 / np.sqrt(C * snr)


def table_from_snr(C_hayward: float, C_bardeen: float):
    """
    Return computed table entries (not hardcoded).
    """
    detectors = [
        ("LIGO O5", 30),
        ("Einstein Telescope", 400),
        ("LISA", 1000),
        ("ngEHT (1%)", 100),
    ]
    rows = []
    for name, snr in detectors:
        rows.append((name, snr, alpha_min_from_snr(C_hayward, snr), alpha_min_from_snr(C_bardeen, snr)))
    return rows