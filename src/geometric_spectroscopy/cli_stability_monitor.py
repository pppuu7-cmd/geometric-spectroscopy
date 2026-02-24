from __future__ import annotations

import sys
from ._cli_utils import run_example_script


def main() -> None:
    # Pass-through to examples/stability_monitor.py
    code = run_example_script("examples/stability_monitor.py", sys.argv[1:])
    raise SystemExit(code)