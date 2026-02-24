import os
import sys
import subprocess
from pathlib import Path


def test_all_metrics_demo_runs_and_writes_csv(tmp_path: Path):
    """
    Integration smoke test:
    - runs examples/all_metrics_demo.py as a subprocess,
    - uses env overrides to keep it fast,
    - checks that 3 CSV summaries are written.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "all_metrics_demo.py"
    assert script.exists()

    results_dir = tmp_path / "results"

    env = os.environ.copy()
    env["GS_RESULTS_DIR"] = str(results_dir)
    env["GS_WKB_ORDER"] = "6"

    # speed knobs
    env["GS_N_R"] = "1400"
    env["GS_R_MAX"] = "40.0"
    env["GS_PROFILE_WIDTH"] = "3.0"
    env["GS_MC_REALIZATIONS"] = "40"

    # NEW: mode-range knobs (major speedup)
    env["GS_ELL_MAX"] = "6"   # ells=2..6
    env["GS_NS_MAX"] = "1"    # ns=(0,1)

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, f"Script failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    expected = [
        results_dir / "hayward_summary.csv",
        results_dir / "bardeen_summary.csv",
        results_dir / "simpson_visser_summary.csv",
    ]
    for p in expected:
        assert p.exists(), f"Missing output: {p}"
        txt = p.read_text(encoding="utf-8", errors="ignore")
        assert "wkb_order" in txt
        assert ",6" in txt
        assert "kappa_raw" in txt