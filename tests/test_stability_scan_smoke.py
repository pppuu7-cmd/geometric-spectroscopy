import numpy as np

from geometric_spectroscopy.stability import stability_scan


def test_stability_scan_smoke_small():
    """
    CI-friendly smoke test for the stability layer.
    We keep the scan intentionally tiny (2 deltas x 1 grid).
    """
    summary = stability_scan(
        model="hayward",
        alpha=0.05,
        L_max=5,
        ells=range(2, 6),
        ns=(0, 1),
        ps_ells=(2, 3),
        order=6,
        deltas=(1e-3, 5e-4),
        grids=(dict(r_min=2.2, r_max=50.0, N_r=2000, profile_width=3.0),),
        theta0=np.zeros(6),
    )

    assert summary.n == 2
    assert np.isfinite(summary.kappa_phys_med)
    assert np.isfinite(summary.kappa_phys_rel_spread)
    assert summary.damped_fraction_min >= 1.0 - 1e-15
    assert summary.max_imag_omega_max < 0.0