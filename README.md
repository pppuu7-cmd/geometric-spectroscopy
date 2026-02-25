
# Geometric Spectroscopy

**Geometric Spectroscopy** is a toolkit designed for **spectral inversion** and **stability diagnostics** to facilitate geometric reconstruction of black hole potential barriers from quasinormal mode (QNM) spectra. This repository provides a fully reproducible implementation of the methods and results presented in the accompanying manuscript:

> *Constructive and Stable Reconstruction of Multipolar Barrier Structure from Black Hole Quasinormal Spectra*

---

## Scientific Scope

This toolkit implements the central inverse resonance map, which takes as input multipolar deformation parameters and produces corresponding quasinormal frequencies:

[
\Theta \longmapsto \omega_{\ell n}
]

where:

* ( \Theta = (\theta_0, \dots, \theta_{L_{\max}}) ) are multipolar deformation parameters,
* ( \omega_{\ell n} ) are the QNM frequencies.

The repository includes numerical routines for verifying:

* **Local injectivity** of the resonance map, i.e., the full rank of the Jacobian matrix.
* **Lipschitz stability** under perturbations in the multipolar deformation space.
* **Effective rank diagnostics**, providing numerical measures of identifiability.
* **Conditioning growth** in relation to the truncation parameter ( L_{\max} ).
* **Bias-dominated reconstruction**, with explicit resolution limits in noisy environments.
* **Regularization-controlled inversion**, based on Tikhonov regularization with L-curve corner selection.

---

## Key Features

* **Conditioning diagnostics**: raw, geometric, and physical condition numbers for Jacobians.
* **Effective rank estimates** based on singular value decomposition.
* **Monte Carlo reconstructions** to assess sensitivity to noise and sampling.
* **Fisher forecast** for parameter precision.
* **Stability monitor** with key checks for:

  * Damped fraction
  * Bounds on the imaginary part of ( \omega )
  * Spread of frequency spectrum
* **Padé resummation** for order stability in WKB expansions.
* **Multipolar expansion diagnostics**, exploring conditioning as a function of ( L_{\max} ).
* **Report bundles**: automatically generated CSV and JSON outputs that summarize key diagnostics.

---

## Installation

To install the toolkit in editable mode:

```bash
pip install -e .
```

### Requirements:

* `numpy`
* `scipy`

Optional (for development and testing):

* `pytest`

Optional (for symbolic manipulations):

* `sympy`

---

## Quick Start (CLI)

After installation, the following commands are available:

### 1) Canonical Model Demo

Run the standard demo for any supported model:

```bash
gs-demo --help
gs-demo --model hayward --mc 0 --Lmax 3
```

Supported models include:

* `hayward`
* `bardeen`
* `simpson_visser`

Optional flags:

* `--mc` enables Monte Carlo sampling.
* `--Lmax` specifies the truncation level for multipolar expansions.
* `--save` writes the results into `results/` directory.

---

### 2) Stability Monitor

Run a compact stability scan:

```bash
gs-stability-monitor
```

This will write output to `results/` directory.

---

### 3) Report Bundle (Strict)

This command produces:

* Per-model summary CSV files
* Stability-monitor diagnostics (CSV and JSON)
* Combined “report bundle” CSV and JSON files

```bash
gs-report-bundle --strict
```

---

### 4) All-Metrics Demo

Generates extracted summaries suitable for publication:

```bash
gs-all-metrics
```

---

## Reproducibility

The repository is fully reproducible, with deterministic seeds used in Monte Carlo simulations to ensure that results can be regenerated independently.

To test the local setup, run the test suite:

```bash
pytest -q
```

CI automatically runs unit tests and smoke tests for the main commands:

* `gs-demo`
* `gs-stability-monitor`
* `gs-report-bundle --strict`

---

## Examples Directory

The repository includes individual scripts for direct execution under `examples/`:

* `examples/model_demo.py`
* `examples/stability_monitor.py`
* `examples/report_bundle.py`
* `examples/all_metrics_demo.py`

While these scripts can be run directly using Python, the recommended interface is through the CLI commands described above for streamlined execution and output generation.

---

## Outputs

By default, the generated outputs are saved to the `results/` directory. This includes:

* CSV tables
* JSON files with summary statistics
* Condition number diagnostics
* Stability and reconstruction results
* Monte Carlo statistics
* Regularization scan outputs

This directory is intended for storing generated artifacts and should not be committed to the repository.

---

## Numerical Stability and Methodology

The core methods are based on the rigorous **inverse resonance framework** introduced in the manuscript. The toolkit ensures:

* **Full rank** of the Jacobian for local injectivity.
* **Lipschitz stability** of the resonance map under perturbations.
* **Controlled conditioning** of the inverse problem, with explicit resolution limits.
* **Numerical identifiability** verified through effective rank measures.
* **Padé resummation** of WKB expansions to stabilize high-order computations.
* **Bias-dominated regimes** in Monte Carlo experiments with systematic scaling analysis.

---

## Citation

If you use this code in academic work, please cite:

* The accompanying manuscript: *Constructive and Stable Reconstruction of Multipolar Barrier Structure from Black Hole Quasinormal Spectra*.
* This repository via `CITATION.cff`.
* The release tag `v0.1.0` for the exact version.

---

## Licensing

### Code License (Dual License: Academic Free / Commercial Paid)

The software is distributed under a dual license:

* **Free for non-commercial academic, educational, or non-profit research**.
* **Commercial use** requires a separate paid license from the author.

Please refer to the `LICENSE` file for details. For commercial licensing inquiries, contact **[pppuuu7@gmail.com](mailto:pppuuu7@gmail.com)**.

### Manuscript / Theory License (CC BY-NC)

The **manuscript content** (text, figures, theory exposition) is licensed under:

* **Creative Commons Attribution–NonCommercial (CC BY-NC)**.

See `MANUSCRIPT_LICENSE.txt` for further details.

---

## Contact

**Aleksey Buyanov**
ORCID: [https://orcid.org/0009-0001-2621-9305](https://orcid.org/0009-0001-2621-9305)
Email: [pppuuu7@gmail.com](mailto:pppuuu7@gmail.com)
Repository: [https://github.com/pppuu7-cmd/geometric-spectroscopy](https://github.com/pppuu7-cmd/geometric-spectroscopy)

---

# Additional Notes for Reproducibility

The toolkit provides full **support for reproducibility**, ensuring that all methods and diagnostics are deterministic when run with the same parameters. The primary focus of this repository is on the controlled inversion and **numerical stability** of the quasinormal mode spectrum, particularly in the context of black hole potential reconstruction. For further details, please refer to the supplementary sections in the manuscript.
