"""Finite-size scaling analysis of sweep results (plan task P1-2).

Consumes the per-L JSON files written by scripts/sweep.py and produces:
  - xi/L crossing points with bootstrap errors, for BOTH chi conventions
    (frozen <M^2>-<|M|>^2 vs Var(M)) — the convention sensitivity check;
  - susceptibility peaks by quadratic fit, and a weighted ln chi_max vs
    ln L fit -> gamma/nu with uncertainty;
  - nu from the scaling collapse;
  - the tau_int table (per L: max and median over the m^2 grid).

Usage:
    python scripts/analyze_fss.py --sweep_dir data/sweep_results_v2 \
        --sizes 16 32 64 --coupling 0.5 \
        --output results/fss_analysis_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qft_graph.analysis.critical import (
    aic_model_average_gamma_nu,
    crossing_with_errors,
    extract_nu,
    fit_gamma_over_nu,
    fit_gamma_over_nu_corrections,
    susceptibility_peak_quadratic,
)
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.run_logging import log_run


def load_sweep(sweep_dir: Path, L: int, coupling: float) -> list[dict]:
    path = sweep_dir / f"sweep_{L}x{L}_lam={coupling}.json"
    with open(path) as f:
        points = json.load(f)
    return sorted(points, key=lambda p: p["m2"])


def main() -> None:
    parser = argparse.ArgumentParser(description="FSS analysis with errors")
    parser.add_argument("--sweep_dir", type=str, default="data/sweep_results_v2")
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--coupling", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="results/fss_analysis_v2.json")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak_dir", type=str, default=None,
                        help="Directory of refined (denser-grid) peak scans; a "
                             "size's chi-peak fit prefers this dir when present. "
                             "Crossings/collapse still use --sweep_dir.")
    parser.add_argument("--peak_n_points", type=int, default=9,
                        help="Points in the quadratic peak-fit window for "
                             "refined scans (denser grid warrants more points).")
    args = parser.parse_args()

    logger = setup_logging()
    sweep_dir = Path(args.sweep_dir)
    sweeps = {L: load_sweep(sweep_dir, L, args.coupling) for L in args.sizes}

    # Refined peak scans (denser grid) used only for the chi-peak / gamma/nu
    # determination; crossings and the nu collapse keep the uniform coarse grid
    peak_sweeps = {}
    if args.peak_dir:
        peak_dir = Path(args.peak_dir)
        for L in args.sizes:
            p = peak_dir / f"sweep_{L}x{L}_lam={args.coupling}.json"
            if p.exists():
                with open(p) as f:
                    peak_sweeps[L] = sorted(json.load(f), key=lambda q: q["m2"])
                logger.info("using refined peak scan for L=%d (%s)", L, p.name)

    report: dict = {"sizes": args.sizes, "coupling": args.coupling}

    # --- xi/L crossings with errors: frozen convention, Var(M) sensitivity
    # check, and the proposed k!=0-only estimator (docs/xi_estimator_issue.md)
    estimators = [
        ("abs", "xi_over_L", "xi_over_L_err"),
        ("var", "xi_var_over_L", "xi_var_over_L_err"),
    ]
    if all("xi_2mom_over_L" in p for pts in sweeps.values() for p in pts):
        estimators.append(("2mom", "xi_2mom_over_L", "xi_2mom_over_L_err"))
    crossings = {conv: [] for conv, _, _ in estimators}
    for conv, key, ekey in estimators:
        for L1, L2 in zip(sorted(args.sizes)[:-1], sorted(args.sizes)[1:]):
            p1, p2 = sweeps[L1], sweeps[L2]
            m2 = np.array([p["m2"] for p in p1])
            m2_2 = np.array([p["m2"] for p in p2])
            if not (len(m2) == len(m2_2) and np.allclose(m2, m2_2)):
                # Mixed-grid inputs (e.g. some sizes refined): crossings
                # require a common grid, so skip this pair rather than fail
                logger.warning("skipping crossing L=%d/%d (m2 grids differ)", L1, L2)
                continue
            center, err = crossing_with_errors(
                m2,
                np.array([p[key] for p in p1]),
                np.array([p[ekey] for p in p1]),
                np.array([p[key] for p in p2]),
                np.array([p[ekey] for p in p2]),
                n_boot=args.n_boot,
                seed=args.seed,
            )
            # A crossing within one grid step of the scan boundary is an
            # edge artifact (noise-level curves touching at the end of the
            # grid), not a resolved crossing — exclude it from m2_c.
            step = abs(m2[1] - m2[0])
            edge = np.isfinite(center) and (
                center < m2.min() + step or center > m2.max() - step
            )
            crossings[conv].append(
                {"pair": [L1, L2], "m2_c": center, "m2_c_err": err,
                 "edge_artifact": bool(edge)}
            )
            logger.info(
                "crossing (%s) L=%d/%d: m2_c = %.4f +/- %.4f%s",
                conv, L1, L2, center, err, "  [EDGE ARTIFACT — excluded]" if edge else "",
            )
    report["crossings"] = crossings

    # Combined m2_c per convention (error-weighted mean over resolved pairs)
    for conv in crossings:
        vals = [
            c for c in crossings[conv]
            if np.isfinite(c["m2_c"]) and not c["edge_artifact"]
        ]
        if vals:
            w = np.array([1.0 / max(c["m2_c_err"], 1e-6) ** 2 for c in vals])
            m2c = float(np.sum(w * [c["m2_c"] for c in vals]) / w.sum())
            m2c_err = float(1.0 / np.sqrt(w.sum()))
            report[f"m2_c_{conv}"] = {"value": m2c, "err": m2c_err}

    if "m2_c_abs" in report and "m2_c_var" in report:
        shift = abs(report["m2_c_abs"]["value"] - report["m2_c_var"]["value"])
        combined_err = np.hypot(report["m2_c_abs"]["err"], report["m2_c_var"]["err"])
        report["convention_sensitivity"] = {
            "m2_c_shift": shift,
            "combined_err": float(combined_err),
            "shift_outside_error": bool(shift > combined_err),
        }
        logger.info(
            "convention sensitivity: |shift| = %.4f vs combined err %.4f (%s)",
            shift, combined_err,
            "OUTSIDE error bar" if shift > combined_err else "within error bar",
        )

    # --- susceptibility peaks (quadratic fit) and gamma/nu ---
    peaks = []
    for L in sorted(args.sizes):
        refined = L in peak_sweeps
        pts = peak_sweeps[L] if refined else sweeps[L]
        res = susceptibility_peak_quadratic(
            np.array([p["m2"] for p in pts]),
            np.array([p["susceptibility"] for p in pts]),
            np.array([p["susceptibility_err"] for p in pts]),
            n_points=args.peak_n_points if refined else 5,
        )
        res["L"] = L
        res["refined"] = refined
        peaks.append(res)
        logger.info(
            "chi peak L=%d%s: m2 = %.4f +/- %.4f, chi_max = %.2f +/- %.2f "
            "(chi2/dof %.2f%s)",
            L, " [refined]" if refined else "",
            res["m2_peak"], res["m2_peak_err"], res["chi_max"], res["chi_max_err"],
            res["chi2_dof"], "" if res["fit_ok"] else " — FIT FAILED, grid max used",
        )
    report["chi_peaks"] = peaks

    L_arr = np.array([p["L"] for p in peaks], dtype=float)
    chi_arr = np.array([p["chi_max"] for p in peaks])
    chi_err_arr = np.array([p["chi_max_err"] for p in peaks])

    gamma_nu, gamma_nu_err = fit_gamma_over_nu(L_arr, chi_arr, chi_err_arr)
    report["gamma_over_nu"] = {"value": gamma_nu, "err": gamma_nu_err, "exact_2d_ising": 1.75}
    logger.info("gamma/nu (naive power law) = %.3f +/- %.3f (exact: 1.75)",
                gamma_nu, gamma_nu_err)

    # --- Robustness cross-checks on the naive power law (Schaich): a
    # corrections-to-scaling fit chi_max = A L^(g/n)(1 + b L^(-w)) and an
    # AIC-weighted average over small-L cuts (arXiv:2008.01069). With clean
    # (unbiased) cluster data these should agree with the naive slope.
    corr_free = fit_gamma_over_nu_corrections(L_arr, chi_arr, chi_err_arr)
    corr_fixed = fit_gamma_over_nu_corrections(L_arr, chi_arr, chi_err_arr, omega=2.0)
    report["gamma_over_nu_corrections"] = {
        "omega_free": corr_free, "omega_fixed_2": corr_fixed,
    }
    if corr_free.get("fit_ok"):
        logger.info(
            "gamma/nu (corrections, omega free) = %.3f +/- %.3f "
            "(omega = %.2f +/- %.2f, chi2/dof %.2f)",
            corr_free["gamma_over_nu"], corr_free["gamma_over_nu_err"],
            corr_free["omega"], corr_free["omega_err"], corr_free["chi2_dof"],
        )
    if corr_fixed.get("fit_ok"):
        logger.info(
            "gamma/nu (corrections, omega=2 fixed) = %.3f +/- %.3f (chi2/dof %.2f)",
            corr_fixed["gamma_over_nu"], corr_fixed["gamma_over_nu_err"],
            corr_fixed["chi2_dof"],
        )
    if len(L_arr) >= 3:
        aic = aic_model_average_gamma_nu(L_arr, chi_arr, chi_err_arr)
        report["gamma_over_nu_aic"] = aic
        logger.info(
            "gamma/nu (AIC model average) = %.3f +/- %.3f (stat %.3f, sys %.3f)",
            aic["gamma_over_nu"], aic["gamma_over_nu_err"],
            aic["stat_err"], aic["sys_err"],
        )

    # --- nu from scaling collapse ---
    # Prefer the k!=0 estimator (defined in both phases); restrict to a
    # window around m2_c and to points where xi was measurable, otherwise
    # the collapse metric is dominated by degenerate xi = 0 entries.
    conv_for_nu = "2mom" if "m2_c_2mom" in report else "abs"
    key_for_nu = {"2mom": "xi_2mom_over_L", "abs": "xi_over_L"}[conv_for_nu]
    m2c_key = f"m2_c_{conv_for_nu}"
    if m2c_key in report:
        m2c = report[m2c_key]["value"]
        window = 0.6  # |m2 - m2_c| window for the collapse fit
        xi_over_L_data = {
            L: [
                (p["m2"], p[key_for_nu])
                for p in sweeps[L]
                if p[key_for_nu] > 0 and abs(p["m2"] - m2c) < window
            ]
            for L in args.sizes
        }
        if all(len(v) >= 4 for v in xi_over_L_data.values()):
            nu, _ = extract_nu(args.sizes, xi_over_L_data, m2c)
            # Honest error: parametric bootstrap over the xi/L point errors
            # AND the m2_c uncertainty (the collapse-quality width badly
            # underestimates it). Points with err > value are dropped inside
            # the resample as unmeasured.
            rng = np.random.default_rng(args.seed)
            err_lookup = {
                (L, p["m2"]): p[f"{key_for_nu}_err"]
                for L in args.sizes for p in sweeps[L]
            }
            m2c_err = report[m2c_key]["err"]
            boots = []
            for _ in range(60):
                m2c_b = m2c + rng.normal(0.0, m2c_err)
                resampled = {}
                ok = True
                for L in args.sizes:
                    pts = []
                    for m2v, y in xi_over_L_data[L]:
                        e = err_lookup[(L, m2v)]
                        yb = y + rng.normal(0.0, e)
                        if yb > 0 and e < max(y, 1e-9):
                            pts.append((m2v, yb))
                    if len(pts) < 4:
                        ok = False
                        break
                    resampled[L] = pts
                if ok:
                    nb, _ = extract_nu(args.sizes, resampled, m2c_b)
                    boots.append(nb)
            nu_err = float(np.std(boots)) if len(boots) > 5 else float("inf")
            report["nu"] = {
                "value": nu, "err": nu_err, "exact_2d_ising": 1.0,
                "estimator": conv_for_nu, "window": window,
                "n_boot": len(boots),
            }
            logger.info("nu = %.3f +/- %.3f (exact: 1, estimator: %s, bootstrap)",
                        nu, nu_err, conv_for_nu)
        else:
            logger.warning("Too few usable xi points for the nu collapse")

    # --- tau_int summary ---
    tau_table = {}
    for L in sorted(args.sizes):
        taus = np.array([p["tau_int"] for p in sweeps[L]])
        i_max = int(np.argmax(taus))
        tau_table[str(L)] = {
            "max": float(taus.max()),
            "median": float(np.median(taus)),
            "argmax_m2": sweeps[L][i_max]["m2"],
            "per_point": [
                {"m2": p["m2"], "tau_int": p["tau_int"], "n_bins": p["n_bins"]}
                for p in sweeps[L]
            ],
        }
        logger.info(
            "tau_int L=%d: median %.2f, max %.2f at m2=%.3f",
            L, tau_table[str(L)]["median"], taus.max(), sweeps[L][i_max]["m2"],
        )
    report["tau_int"] = tau_table

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Analysis written to %s", out_path)

    log_run(
        out_path.stem + "_provenance",
        config={
            "sweep_dir": str(sweep_dir),
            "sizes": args.sizes,
            "coupling": args.coupling,
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        metrics={
            k: report[k]
            for k in ("m2_c_abs", "m2_c_var", "m2_c_2mom", "gamma_over_nu",
                      "nu", "convention_sensitivity")
            if k in report
        },
        extra={"analysis_json": str(out_path)},
    )


if __name__ == "__main__":
    main()
