# src/geometric_spectroscopy/__init__.py
"""Geometric Spectroscopy package.

Public API is intentionally small and stable.
"""

from .resonances import compute_qnms, compute_data_vector
from .jacobian import (
    build_jacobian,
    conditioning_raw,
    conditioning_geom,
    conditioning_phys,
    deformation_weights_phys,
    effective_rank,
)
from .monte_carlo import fisher_cov_theta, fisher_sigma_theta, monte_carlo_theta
from .stability import (
    StabilityPoint,
    StabilitySummary,
    stability_point,
    stability_scan,
    format_stability_summary,
    assert_stability,
)

__all__ = [
    "compute_qnms",
    "compute_data_vector",
    "build_jacobian",
    "conditioning_raw",
    "conditioning_geom",
    "conditioning_phys",
    "deformation_weights_phys",
    "effective_rank",
    "fisher_cov_theta",
    "fisher_sigma_theta",
    "monte_carlo_theta",
    "StabilityPoint",
    "StabilitySummary",
    "stability_point",
    "stability_scan",
    "format_stability_summary",
    "assert_stability",
]

# Keep in sync with pyproject.toml / package metadata.
__version__ = "0.1.0"