import numpy as np

from geometric_spectroscopy.potentials import build_potential_rstar
from geometric_spectroscopy.qnm_wkb import qnm_wkb


def test_wkb6_smoke_contract_hayward_damped_and_deterministic():
    """
    Standard-B smoke contract for the WKB6 pipeline:

      model -> V(r*) -> peak_derivatives(k<=6) -> WKB6 + Padé[3/3] -> omega

    We require:
      - omega is finite complex
      - Im(omega) < 0 (damped mode)
      - deterministic under repeated identical calls (within tight tolerance)
    """
    rstar, V, _weights = build_potential_rstar(
        model="hayward",
        ell=2,
        alpha=0.05,
        theta=np.zeros(6),
        L_max=5,
        r_min=2.2,
        r_max=50.0,
        N_r=2200,
        profile_width=3.0,
    )

    res1 = qnm_wkb(rstar, V, n=0, order=6, ell=2)
    res2 = qnm_wkb(rstar, V, n=0, order=6, ell=2)

    w1 = complex(res1.omega)
    w2 = complex(res2.omega)

    assert np.isfinite(w1.real) and np.isfinite(w1.imag)
    assert w1.imag < 0.0

    # Determinism: should be extremely tight for a pure deterministic pipeline.
    # Keep small tolerance to avoid platform FP noise without hiding real regressions.
    assert abs(w1.real - w2.real) <= 1e-10
    assert abs(w1.imag - w2.imag) <= 1e-10