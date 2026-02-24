from __future__ import annotations

import sys
from ._cli_utils import run_example_script


def main() -> None:
    # Pass-through to examples/report_bundle.py
    code = run_example_script("examples/report_bundle.py", sys.argv[1:])
    raise SystemExit(code)