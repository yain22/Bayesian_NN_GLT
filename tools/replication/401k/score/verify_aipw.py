"""Independently verify the reported five-fold AIPW estimates using NumPy.

Usage: python verify_aipw.py data.csv predictions.csv.gz results_directory
"""
import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def key(row):
    return int(row["depth"]), int(row["width"]), row["method"]


def check_values(reported, calculated):
    for name, value in calculated.items():
        np.testing.assert_allclose(float(reported[name]), value, rtol=1e-11, atol=1e-8,
                                   err_msg=f"{name} differs")


def verify(data_path, predictions, output):
    rows = sorted(read_rows(data_path), key=lambda r: int(r["row_id"]))
    index = {int(r["row_id"]): i for i, r in enumerate(rows)}
    n = len(index)
    assert len(rows) == n == 9915
    y = np.array([float(r["net_tfa"]) for r in rows])
    d = np.array([int(r["e401"]) for r in rows])
    folds = np.array([int(r["fold"]) for r in rows])
    by_group = defaultdict(lambda: np.full((n, 3), np.nan))
    for r in read_rows(predictions):
        i = index[int(r["row_id"])]
        assert int(r["fold"]) == folds[i]
        nuisance = by_group[key(r)]
        assert np.isnan(nuisance[i, 0]), "Duplicate held-out prediction"
        nuisance[i] = [float(r[x]) for x in ("m0_hat", "m1_hat", "e_hat")]
    assert len(by_group) == 36
    diagnostics = {key(r): r for r in read_rows(output / "nuisance_diagnostics.csv")}
    estimates = {(key(r), float(r["bound"])): r for r in read_rows(output / "aipw_estimates.csv")}
    assert len(diagnostics) == 36 and len(estimates) == 108
    for group, nuisance in by_group.items():
        assert np.isfinite(nuisance).all()
        m0, m1, raw = nuisance.T
        assert (raw >= 0).all() and (raw <= 1).all()
        residual = y - np.where(d, m1, m0)
        low, high = np.quantile(raw, [0.01, 0.99])
        check_values(diagnostics[group], dict(
            n=n, n_treated=d.sum(), outcome_rmse=np.sqrt(np.mean(residual**2)),
            outcome_rmse_0=np.sqrt(np.mean(residual[d == 0]**2)),
            outcome_rmse_1=np.sqrt(np.mean(residual[d == 1]**2)),
            propensity_brier=np.mean((raw-d)**2), propensity_min=raw.min(),
            propensity_max=raw.max(), propensity_p01=low, propensity_p99=high
        ))
        for threshold in (0.01, 0.025, 0.05):
            e = np.clip(raw, threshold, 1-threshold)
            mu1 = m1 + d / e * (y - m1)
            mu0 = m0 + (1-d) / (1-e) * (y - m0)
            theta = np.mean(mu1) - np.mean(mu0)
            centered_difference = (mu1-mu1.mean()) - (mu0-mu0.mean())
            se = np.linalg.norm(centered_difference) / n
            width = NormalDist().inv_cdf(0.975) * se
            check_values(estimates[group, threshold], dict(
                ate=theta, se=se, ci_low=theta-width, ci_high=theta+width,
                n_clip_low=np.sum(raw < threshold), n_clip_high=np.sum(raw > 1-threshold)
            ))
    print("Verified 108 pooled estimates, standard errors, intervals and clipping counts, "
          "and all 36 nuisance diagnostics from raw held-out predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.data, args.predictions, args.output)
