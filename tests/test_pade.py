import math
import numpy as np

from geometric_spectroscopy.pade import pade_eval, pade_ensemble


def test_pade_11_exact_for_geometric_series():
    # f(x)=1/(1-x)=sum x^k, c_k=1
    coeffs = [1.0 + 0j] * 6
    pr = pade_eval(coeffs, m=1, n=1, eps=0.3)
    assert abs(pr.value - (1.0 / (1.0 - 0.3))) < 1e-12
    assert pr.stable


def test_pade_33_reasonable_for_exp():
    # exp(x) series: sum x^k/k!
    coeffs = [1.0 / math.factorial(k) for k in range(7)]
    pr = pade_eval(coeffs, m=3, n=3, eps=1.0)
    # [3/3] Padé for exp(1) is quite accurate
    assert abs(pr.value - np.e) < 5e-3
    assert pr.stable


def test_pade_pole_diagnostics_detects_near_pole():
    # f(x)=1/(1-x) has a pole at x=1. Padé [1/1] reproduces it exactly.
    coeffs = [1.0 + 0j] * 4
    pr = pade_eval(coeffs, m=1, n=1, eps=0.99, pole_guard=0.2)
    # pole is at 1 -> distance about 0.01 < 0.2 => unstable
    assert pr.pole_distance < 0.2
    assert pr.stable is False


def test_pade_ensemble_returns_mean_and_std():
    coeffs = [1.0 / math.factorial(k) for k in range(9)]  # enough for [4/4]
    out = pade_ensemble(coeffs, approximants=[(2, 2), (3, 3), (4, 4)], eps=1.0)
    assert out["total_count"] == 3
    assert out["stable_count"] >= 2
    assert out["mean_value"] is not None
    assert out["std_value"] is not None