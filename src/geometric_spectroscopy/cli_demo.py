from __future__ import annotations

import sys
from ._cli_utils import run_example_script


def main() -> None:
    """
    Entry point for `gs-demo`.
    Pass-through to examples/model_demo.py.
    """
    code = run_example_script("examples/model_demo.py", sys.argv[1:])
    raise SystemExit(code)