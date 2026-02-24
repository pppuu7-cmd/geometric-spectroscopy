from __future__ import annotations

import sys
from ._cli_utils import run_example_script


def main() -> None:
    # Pass-through to examples/all_metrics_demo.py
    code = run_example_script("examples/all_metrics_demo.py", sys.argv[1:])
    raise SystemExit(code)