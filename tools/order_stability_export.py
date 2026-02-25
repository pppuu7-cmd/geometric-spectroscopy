import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[warn] no rows for {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {path} ({len(rows)} rows)")


def try_import(module_path: str):
    __import__(module_path)
    import sys

    return sys.modules[module_path]


# ---------------------------------------------------------
# Project-specific adapters
# ---------------------------------------------------------
@dataclass
class CaseSpec:
    name: str
    model_kwargs: Dict[str, Any]


def get_default_cases() -> List[CaseSpec]:
    """
    В potentials.py у тебя:
      - hayward с alpha=0 эквивалентен Schwarzschild
    Поэтому для базового бенчмарка используем его.
    """
    return [
        CaseSpec(name="schwarzschild_via_hayward", model_kwargs={"model": "hayward", "alpha": 0.0}),
        # Можно добавить ещё кейсы (раскомментируй при необходимости):
        # CaseSpec(name="bardeen", model_kwargs={"model": "bardeen", "alpha": 0.6}),
        # CaseSpec(name="hayward", model_kwargs={"model": "hayward", "alpha": 0.6}),
        # CaseSpec(name="simpson_visser", model_kwargs={"model": "simpson_visser", "alpha": 0.6}),
    ]


def compute_qnms(case: CaseSpec, wkb_order: int, use_pade: bool, N: int = 6) -> List[complex]:
    """
    Считает комплексные QNM частоты omega для n=0..N-1.
    Поддерживаем 2 режима:
      - order=3 -> qnm_wkb3
      - order=6 -> qnm_wkb6 с pade True/False
    """
    pot = try_import("geometric_spectroscopy.potentials")
    qnm = try_import("geometric_spectroscopy.qnm_wkb")

    model = str(case.model_kwargs["model"])
    alpha = float(case.model_kwargs.get("alpha", 0.0))

    # Параметры эксперимента (можно менять под статью)
    ell = 2
    L_max = 6
    theta = np.zeros(L_max + 1, dtype=float)

    # Строим потенциал на r* сетке
    rstar, V, _w = pot.build_potential_rstar(
        model=model,
        ell=ell,
        alpha=alpha,
        theta=theta,
        L_max=L_max,
        N_r=6000,
        r_min=2.2,
        r_max=60.0,
    )

    omegas: List[complex] = []
    for n in range(int(N)):
        if int(wkb_order) == 3:
            res = qnm.qnm_wkb3(rstar, V, ell=ell, n=n)
            omegas.append(complex(res.omega))
        elif int(wkb_order) == 6:
            res = qnm.qnm_wkb6(
                rstar,
                V,
                ell=ell,
                n=n,
                pade=bool(use_pade),
                return_diagnostics=False,
            )
            omegas.append(complex(res.omega))
        else:
            raise ValueError("Supported orders in this script: 3, 6")

    return omegas


def compute_jacobian_svd(case: CaseSpec, wkb_order: int, use_pade_label: bool, N: int = 6) -> Dict[str, Any]:
    """
    Строит Якобиан J и считает SVD-диагностики:
      rank, sigma_min, sigma_max, cond.

    ВАЖНО:
      Текущая реализация build_jacobian(...) использует compute_data_vector(),
      который внутри вызывает qnm_wkb() с фиксированной логикой (order=6 включает pade=True).
      Поэтому здесь use_pade_label — только метка, а сравнение делаем главным образом по order=3 vs order=6.
    """
    jac = try_import("geometric_spectroscopy.jacobian")

    model = str(case.model_kwargs["model"])
    alpha = float(case.model_kwargs.get("alpha", 0.0))

    L_max = 6

    # Чтобы было быстрее и всё равно показательно:
    ells = range(2, 6)  # 2..5
    ns = (0, 1, 2)

    Jc = jac.build_jacobian(
        model=model,
        alpha=alpha,
        L_max=L_max,
        ells=ells,
        ns=ns,
        order=int(wkb_order),
        delta=1e-3,
        theta0=None,
        grid_kwargs=None,
        ps_ells=(2, 3),
    )

    Jr = jac.realify_complex_residuals(Jc)
    s = np.linalg.svd(Jr, compute_uv=False)

    smax = float(s[0])
    smin = float(s[-1])
    cond = float(smax / smin) if smin > 0 else float("inf")

    # эффективный ранг (устойчиво для численной статьи)
    rank = int(np.sum(s > 1e-12 * smax))

    return {
        "case": case.name,
        "order": int(wkb_order),
        "pade_label": int(bool(use_pade_label)),
        "rank": rank,
        "sigma_max": smax,
        "sigma_min": smin,
        "cond": cond,
        "ells_min": 2,
        "ells_max": 5,
        "ns": "0,1,2",
        "L_max": L_max,
    }


# ---------------------------------------------------------
# Main export
# ---------------------------------------------------------
def main() -> None:
    out_dir = os.path.join("artifacts", "order_stability")
    ensure_dir(out_dir)

    cases = get_default_cases()

    # Здесь сравниваем order=3 и order=6 (это уже сильный научный результат)
    orders = [3, 6]

    # Для частот: order=6 считаем и без Padé, и с Padé
    pade_flags = [False, True]

    # число мод (n=0..N-1)
    N = 6

    freq_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    svd_rows: List[Dict[str, Any]] = []

    for case in cases:
        for use_pade in pade_flags:
            prev_omegas = None
            for p in orders:
                omegas = compute_qnms(case, p, use_pade, N=N)

                # частоты
                for i, w in enumerate(omegas, start=0):
                    freq_rows.append(
                        {
                            "case": case.name,
                            "pade": int(use_pade),
                            "order": int(p),
                            "n": int(i),
                            "omega_re": float(w.real),
                            "omega_im": float(w.imag),
                        }
                    )

                    # приращения по порядку
                    if prev_omegas is not None and i < len(prev_omegas):
                        dw = abs(w - prev_omegas[i])
                        delta_rows.append(
                            {
                                "case": case.name,
                                "pade": int(use_pade),
                                "order": int(p),
                                "n": int(i),
                                "delta_abs": float(dw),
                            }
                        )

                prev_omegas = omegas

                # SVD Якобиана (Padé здесь метка)
                svd_rows.append(compute_jacobian_svd(case, p, use_pade_label=use_pade, N=N))

    write_csv(os.path.join(out_dir, "freq_by_order.csv"), freq_rows)
    write_csv(os.path.join(out_dir, "delta_by_order.csv"), delta_rows)
    write_csv(os.path.join(out_dir, "jacobian_svd_by_order.csv"), svd_rows)

    print("\n[done] CSV artifacts ready:")
    print(f"  - {os.path.join(out_dir, 'freq_by_order.csv')}")
    print(f"  - {os.path.join(out_dir, 'delta_by_order.csv')}")
    print(f"  - {os.path.join(out_dir, 'jacobian_svd_by_order.csv')}")
    print("\nSuggested plots:")
    print("  1) delta_abs vs order (log y) for each n, raw vs pade (for order=6)")
    print("  2) sigma_min vs order (3 vs 6) and cond vs order")
    print("  3) omega_re/omega_im vs order (3 vs 6) and pade effect at order=6")


if __name__ == "__main__":
    main()