import numpy as np
from geometric_spectroscopy.jacobian import build_jacobian

def test_full_rank():
    J = build_jacobian(
        model="hayward",
        alpha=0.05,
        L_max=5,
        delta=1e-3,
        ells=range(2, 13),
        ns=(0, 1, 2),
        ps_ells=(2, 3),
    )

    Jr = np.vstack([J.real, J.imag])
    s = np.linalg.svd(Jr, compute_uv=False)

    rank = np.sum(s > 1e-6 * s[0])
    assert rank == 6