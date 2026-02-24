from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _repo_root_from_package_file(package_file: str) -> Path:
    """
    Resolve repo root assuming layout:
      <repo>/src/geometric_spectroscopy/<this file>
    """
    p = Path(package_file).resolve()
    # .../src/geometric_spectroscopy/_cli_utils.py -> parents[2] == .../src
    # repo root is parents[3] if src/ is directly under repo.
    # But on some layouts it may differ; we'll search upwards for pyproject.toml.
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume typical src-layout
    return p.parents[3]


def run_example_script(example_relpath: str, argv_tail: list[str]) -> int:
    """
    Run examples/<script>.py as if executed directly with Python.
    """
    repo_root = _repo_root_from_package_file(__file__)
    script_path = repo_root / example_relpath

    if not script_path.exists():
        sys.stderr.write(
            f"[geometric-spectroscopy] Cannot find '{example_relpath}'.\n"
            f"Expected at: {script_path}\n"
            f"This CLI works when installed from the repository (editable install).\n"
        )
        return 2

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path)] + argv_tail
        runpy.run_path(str(script_path), run_name="__main__")
        return 0
    except SystemExit as e:
        # propagate exit code from argparse scripts
        code = int(e.code) if isinstance(e.code, int) else 1
        return code
    finally:
        sys.argv = old_argv