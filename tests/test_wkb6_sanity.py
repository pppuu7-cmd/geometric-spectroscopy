import numpy as np

from geometric_spectroscopy.potentials import build_potential_rstar
from geometric_spectroscopy.qnm_wkb import qnm_wkb


def test_wkb_returns_finite_and_damped():
    rstar, V, _weights = build_potential_rstar(
        model="hayward",
        ell=2,
        alpha=0.05,
        theta=np.zeros(6),
        L_max=5,
        r_min=2.2,
        r_max=50.0,
        N_r=2500,
    )

    res = qnm_wkb(rstar, V, n=0, order=6, ell=2)
    w = complex(res.omega)

    assert np.isfinite(w.real)
    assert np.isfinite(w.imag)
    assert w.imag < 0.0


def test_wkb6_stability_on_grid():
    common = dict(
        model="hayward",
        alpha=0.05,
        theta=np.zeros(6),
        L_max=5,
        r_min=2.2,
        r_max=50.0,
    )

    r1, V1, _ = build_potential_rstar(ell=2, N_r=2000, **common)
    r2, V2, _ = build_potential_rstar(ell=2, N_r=3000, **common)

    w1 = complex(qnm_wkb(r1, V1, n=0, order=6, ell=2).omega)
    w2 = complex(qnm_wkb(r2, V2, n=0, order=6, ell=2).omega)

    # Sanity: should be close-ish under grid refinement (not ultra strict)
    assert abs(w1.real - w2.real) < 0.05
    assert abs(w1.imag - w2.imag) < 0.05