# Geometric Spectroscopy

[![CI](https://github.com/pppuu7-cmd/geometric-spectroscopy/actions/workflows/ci.yml/badge.svg)](https://github.com/pppuu7-cmd/geometric-spectroscopy/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![Version](https://img.shields.io/badge/version-0.1.0-informational)

**Geometric Spectroscopy** is a research-grade toolkit for *spectral inversion* and *stability diagnostics* aimed at geometric reconstruction from quasinormal mode (QNM) spectra.

Key features:
- **Conditioning diagnostics**: raw / geometric / physical condition numbers for Jacobians.
- **Effective rank** estimates (numerical identifiability).
- **Fisher forecasts** and **Monte Carlo** reconstructions for deformation parameters.
- **Stability monitor** with guardrail checks (damped fraction, Im ω bounds, spread metrics).
- **Report bundles**: one-shot CSV+JSON outputs suitable for paper supplements and lab logs.

---

## Installation

Editable install from the repository:

```bash
pip install -e .
````

Requirements are minimal:

* `numpy`
* `scipy`

(For development/testing: `pytest`.)

---

## Quick start (CLI)

After installation, the following commands are available:

### 1) Canonical model demo

Runs the standard demo for any supported model.

```bash
gs-demo --help
gs-demo --model hayward --mc 0 --Lmax 3
```

Supported models:

* `hayward`
* `bardeen`
* `simpson_visser`

Optional flags:

* `--save` writes a small CSV+JSON bundle into `results/`.

### 2) Stability monitor

Runs a compact stability scan and writes CSV+JSON into `results/`:

```bash
gs-stability-monitor
```

### 3) Report bundle (strict)

Produces:

* per-model summary CSV files
* stability-monitor CSV+JSON
* a combined “report bundle” CSV+JSON

```bash
gs-report-bundle --strict
```

### 4) All-metrics demo (extracted summaries)

```bash
gs-all-metrics
```

---

## Outputs

By default, examples and CLIs write outputs into:

* `results/`

This directory is intended for *generated artifacts* and should not be committed.

---

## Reproducibility

Local test suite:

```bash
pytest -q
```

CI runs:

* unit tests
* smoke checks for `gs-demo`, `gs-stability-monitor`, and `gs-report-bundle --strict`

---

## Examples directory

The repository also contains direct scripts under `examples/`:

* `examples/model_demo.py`
* `examples/stability_monitor.py`
* `examples/report_bundle.py`
* `examples/all_metrics_demo.py`

You can run them directly with `python examples/...`, but the recommended interface is via the CLI commands above.

---

## Citation

If you use this code in academic work, please cite it via `CITATION.cff`.

(You can also cite the repository URL and release tag `v0.1.0`.)

---

## Licensing

### Code license (Dual License: Academic Free / Commercial Paid)

The **software code** is distributed under a **dual license**:

* **Free** for non-commercial academic / educational / non-profit research use.
* **Commercial use** requires a separate paid license from the author.

See: `LICENSE`
Commercial licensing: **[pppuuu7@gmail.com](mailto:pppuuu7@gmail.com)**

### Manuscript / theory license (CC BY-NC)

The **paper/manuscript content** (text, figures, theory exposition) is licensed separately under:

* **Creative Commons Attribution–NonCommercial (CC BY-NC)**

See: `MANUSCRIPT_LICENSE.txt`

---

## Contact

Aleksey Buyanov
ORCID: [https://orcid.org/0009-0001-2621-9305](https://orcid.org/0009-0001-2621-9305)
Email: [pppuuu7@gmail.com](mailto:pppuuu7@gmail.com)
Repository: [https://github.com/pppuu7-cmd/geometric-spectroscopy](https://github.com/pppuu7-cmd/geometric-spectroscopy)
