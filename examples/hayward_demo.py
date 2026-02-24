#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    script = here / "model_demo.py"
    cmd = [sys.executable, str(script), "--model", "hayward"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()