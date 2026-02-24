#!/usr/bin/env python3
"""
One-shot "report bundle" runner (Standard B, hardened).

Runs:
  - examples/all_metrics_demo.py
  - examples/stability_monitor.py

Then bundles:
  - One combined CSV (tabular)
  - One combined JSON ("report bundle")

Hardening:
  - Fails fast if expected artifacts are missing / empty.
  - Prevents "silent partial bundle".
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RESULTS_DIRNAME = "results"


@dataclass(frozen=True)
class ProducedArtifacts:
    csv_paths: List[Path]
    json_paths: List[Path]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parent.parent


def _results_dir(repo_root: Path) -> Path:
    d = repo_root / RESULTS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_script(repo_root: Path, rel_path: str, extra_args: Optional[List[str]] = None) -> None:
    script = repo_root / rel_path
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)

    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _latest_by_prefix(results_dir: Path, prefix: str, suffix: str) -> Optional[Path]:
    cands = sorted(results_dir.glob(f"{prefix}*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _parse_summary_csv_kv(path: Path) -> Dict[str, Any]:
    """
    Parses *_summary.csv produced by all_metrics_demo.
    Format is not tabular; it's key,value lines with blank lines and sections.
    We extract:
      - model, alpha_true, L_max, delta, wkb_order, sigma_omega, ells, ns, ps_ells,
        N_r, r_max, profile_width, mc_realizations,
        kappa_raw, kappa_geom, kappa_phys, effective_rank
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    out: Dict[str, Any] = {}

    def _maybe_num(s: str) -> Any:
        ss = s.strip()
        if ss == "":
            return ss
        try:
            if "." in ss or "e" in ss.lower():
                return float(ss)
            return int(ss)
        except Exception:
            return ss.strip('"')

    for ln in lines:
        if not ln.strip():
            continue
        if ln.strip() in {"singular_values", "param,theta_true,fisher_sigma,mc_mean,mc_std"}:
            break
        parts = ln.split(",", 1)
        if len(parts) != 2:
            continue
        k, v = parts[0].strip(), parts[1].strip()
        out[k] = _maybe_num(v)

    if "model" in out and isinstance(out["model"], str):
        out["model"] = out["model"].strip()

    return out


def _assert_file_nonempty(path: Path, what: str) -> None:
    if not path.exists():
        raise RuntimeError(f"[bundle] Missing {what}: {path}")
    try:
        size = path.stat().st_size
    except Exception:
        size = 0
    if size <= 0:
        raise RuntimeError(f"[bundle] Empty {what}: {path}")


def _assert_tabular_csv_has_rows(path: Path, what: str) -> None:
    _assert_file_nonempty(path, what)
    fields, rows = _read_csv_rows(path)
    if not fields:
        raise RuntimeError(f"[bundle] {what} has no header/columns: {path}")
    if len(rows) == 0:
        raise RuntimeError(f"[bundle] {what} has 0 data rows: {path}")


def _extract_all_metrics_demo(results_dir: Path, tag: str, ts: str) -> ProducedArtifacts:
    """
    Collects *_summary.csv and converts them to:
      - results/all_metrics_demo_extracted_<tag>_<ts>.csv  (tabular)
      - results/all_metrics_demo_extracted_<tag>_<ts>.json (structured)
    Returns those paths as artifacts.
    """
    summary_paths = sorted(results_dir.glob("*_summary.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    wanted = {"hayward_summary.csv", "bardeen_summary.csv", "simpson_visser_summary.csv"}
    picked: List[Path] = [p for p in summary_paths if p.name in wanted]
    if len(picked) == 0 and len(summary_paths) > 0:
        picked = summary_paths[:3]

    rows: List[Dict[str, Any]] = []
    for p in picked:
        rows.append(_parse_summary_csv_kv(p))

    out_csv = results_dir / f"all_metrics_demo_extracted_{tag}_{ts}.csv"
    out_json = results_dir / f"all_metrics_demo_extracted_{tag}_{ts}.json"

    cols: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)

    preferred = [
        "model",
        "wkb_order",
        "alpha_true",
        "L_max",
        "delta",
        "N_r",
        "r_max",
        "profile_width",
        "mc_realizations",
        "kappa_raw",
        "kappa_geom",
        "kappa_phys",
        "effective_rank",
        "ps_ells",
        "ns",
        "ells",
        "sigma_omega",
    ]
    ordered_cols = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]

    _write_csv(out_csv, ordered_cols, rows)

    payload = {
        "extracted_from": [str(p) for p in picked],
        "rows": rows,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return ProducedArtifacts(csv_paths=[out_csv], json_paths=[out_json])


def _merge_csvs(sections: List[Tuple[str, Path]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Merge multiple *tabular* CSVs by stacking rows and adding report_section.
    sections: list of (section_name, csv_path)
    """
    all_fields: List[str] = ["report_section"]
    all_rows: List[Dict[str, Any]] = []

    for section, csv_path in sections:
        fields, rows = _read_csv_rows(csv_path)
        for fn in fields:
            if fn not in all_fields:
                all_fields.append(fn)
        for r in rows:
            rr: Dict[str, Any] = dict(r)
            rr["report_section"] = section
            all_rows.append(rr)

    return all_fields, all_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hayward", help="Tag added to bundle filename (default: hayward)")
    ap.add_argument("--skip-all-metrics", action="store_true")
    ap.add_argument("--skip-stability", action="store_true")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected artifact is missing (recommended; CI should use this).",
    )
    args = ap.parse_args()

    repo_root = _repo_root_from_this_file()
    results_dir = _results_dir(repo_root)

    if not args.skip_all_metrics:
        _run_script(repo_root, "examples/all_metrics_demo.py")
    if not args.skip_stability:
        _run_script(repo_root, "examples/stability_monitor.py")

    ts = _timestamp()

    # all_metrics_demo: extract *_summary.csv -> tabular artifacts
    all_metrics = ProducedArtifacts(csv_paths=[], json_paths=[])
    if not args.skip_all_metrics:
        all_metrics = _extract_all_metrics_demo(results_dir, args.tag, ts)
        if args.strict:
            _assert_tabular_csv_has_rows(all_metrics.csv_paths[0], "all_metrics_demo extracted CSV")
            _assert_file_nonempty(all_metrics.json_paths[0], "all_metrics_demo extracted JSON")

    # stability_monitor: use latest standard artifacts
    stability_csv = _latest_by_prefix(results_dir, "stability_monitor_", ".csv")
    stability_json = _latest_by_prefix(results_dir, "stability_monitor_", ".json")
    stability = ProducedArtifacts(
        csv_paths=[stability_csv] if stability_csv else [],
        json_paths=[stability_json] if stability_json else [],
    )
    if args.strict and not args.skip_stability:
        if not stability.csv_paths:
            raise RuntimeError("[bundle] Missing stability_monitor CSV (no files with prefix stability_monitor_*.csv)")
        if not stability.json_paths:
            raise RuntimeError("[bundle] Missing stability_monitor JSON (no files with prefix stability_monitor_*.json)")
        _assert_tabular_csv_has_rows(stability.csv_paths[0], "stability_monitor CSV")
        _assert_file_nonempty(stability.json_paths[0], "stability_monitor JSON")

    # Bundle JSON
    bundle: Dict[str, Any] = {
        "bundle_version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "artifacts": {
            "all_metrics_demo": {
                "csv": [str(p) for p in all_metrics.csv_paths],
                "json": [str(p) for p in all_metrics.json_paths],
            },
            "stability_monitor": {
                "csv": [str(p) for p in stability.csv_paths],
                "json": [str(p) for p in stability.json_paths],
            },
        },
        "sections": {},
    }

    if all_metrics.json_paths:
        bundle["sections"]["all_metrics_demo"] = _read_json(all_metrics.json_paths[0])
    if stability.json_paths:
        bundle["sections"]["stability_monitor"] = _read_json(stability.json_paths[0])

    # Combined CSV
    csv_sections: List[Tuple[str, Path]] = []
    for p in all_metrics.csv_paths:
        csv_sections.append(("all_metrics_demo", p))
    for p in stability.csv_paths:
        csv_sections.append(("stability_monitor", p))

    merged_fields, merged_rows = _merge_csvs(csv_sections)

    out_csv = results_dir / f"report_bundle_{args.tag}_{ts}.csv"
    out_json = results_dir / f"report_bundle_{args.tag}_{ts}.json"

    _write_csv(out_csv, merged_fields, merged_rows)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    if args.strict:
        _assert_tabular_csv_has_rows(out_csv, "report_bundle CSV")
        _assert_file_nonempty(out_json, "report_bundle JSON")

    print("\nReport bundle saved:")
    print(f"  {out_csv}")
    print(f"  {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())