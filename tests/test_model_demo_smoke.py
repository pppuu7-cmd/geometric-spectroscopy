import subprocess
import sys
from pathlib import Path


def _run_demo(model: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "model_demo.py"

    cmd = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--mc",
        "0",          # skip Monte Carlo for fast CI
        "--Lmax",
        "3",          # small size for speed
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Geometric Spectroscopy Demo" in result.stdout


def test_model_demo_hayward():
    _run_demo("hayward")


def test_model_demo_bardeen():
    _run_demo("bardeen")


def test_model_demo_simpson_visser():
    _run_demo("simpson_visser")