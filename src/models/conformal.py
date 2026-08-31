"""S5 — conformal calibration and the out-of-distribution guard (plan §5.4).

Split-conformal prediction gives distribution-free coverage guarantees with no
assumptions about the model, which is what a sample this small needs. The
empirical 80% band from the raw quantile models covers only ~70% of held-out
districts; conformal widening fixes that and makes the coverage a checkable
claim rather than an aspiration.

The Mahalanobis guard handles the other failure mode: a district unlike
anything in training should get rule-based advice, clearly labelled, rather
than a confident guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src import config
from src.eval.splits import group_kfold


class SplitConformal:
    """Distribution-free interval calibration from held-out nonconformity scores."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_: float | None = None

    def calibrate(self, y_true, lower, upper) -> "SplitConformal":
        """Fit the interval half-width from a calibration fold.

        The nonconformity score is how far outside the model's own band the
        truth fell — negative when it fell inside. The (1-alpha) quantile of
        those scores is how much the band must grow.
        """
        y = np.asarray(y_true, dtype=float)
        lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        scores = np.maximum(lo - y, y - hi)
        n = len(scores)
        # the finite-sample corrected quantile level
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(scores, level, method="higher"))
        return self

    def apply(self, lower, upper) -> tuple[np.ndarray, np.ndarray]:
        if self.q_ is None:
            raise RuntimeError("calibrate() first")
        return np.asarray(lower, dtype=float) - self.q_, np.asarray(upper, dtype=float) + self.q_


def cross_val_conformal(
    df: pd.DataFrame,
    lower_col: str,
    upper_col: str,
    truth_col: str = "yield_z",
    alpha: float = 0.1,
    n_splits: int = config.N_FOLDS,
) -> pd.DataFrame:
    """Calibrate and apply conformally, grouped by district throughout.

    Each fold calibrates on the *other* districts, so no district ever
    calibrates its own interval.
    """
    lo = np.full(len(df), np.nan)
    hi = np.full(len(df), np.nan)
    for train_idx, test_idx in group_kfold(df, n_splits):
        cal = df.iloc[train_idx]
        sc = SplitConformal(alpha).calibrate(cal[truth_col], cal[lower_col], cal[upper_col])
        l, u = sc.apply(df.iloc[test_idx][lower_col], df.iloc[test_idx][upper_col])
        lo[test_idx], hi[test_idx] = l, u
    return pd.DataFrame({f"conf_lo_{int((1-alpha)*100)}": lo,
                         f"conf_hi_{int((1-alpha)*100)}": hi}, index=df.index)


class MahalanobisGuard:
    """Out-of-distribution detector over the training feature distribution.

    Beyond the threshold the learned score is suppressed and the S2 rule score
    is served alone, labelled as such.

    Two things make this actually discriminate, rather than being decorative:

    **It is fitted at taluka grain, not district grain.** The first version was
    fitted on the same 34 district vectors it then scored, so nothing was ever
    out-of-distribution and the guard fired on 0 of 30 talukas. Fitting on the
    351 talukas that the districts are built from gives a distribution with
    real spread, and a query taluka can genuinely sit outside it.

    **The threshold is empirical, not asymptotic.** The chi-square quantile
    assumes multivariate normality over p features; with shrunk covariance and
    heavily skewed agro-climatic variables that assumption is badly wrong and
    produced a threshold no real taluka could reach. The cut is a quantile of
    the training distances, so it is calibrated to flag a stated fraction of
    the training set and holds its meaning out of sample.

    **What the threshold should be at serving time.** Every taluka this system
    serves is one of the 351 the guard is fitted on, so by construction none of
    them is novel — and a guard that abstains on Kolhapur, a well-covered
    district, is not being cautious but simply wrong. ``quantile=1.0`` sets the
    cut at the largest distance in the corpus: in-corpus queries never abstain,
    and anything genuinely outside it does. Evidence that this still
    discriminates comes from a regional holdout rather than from the serving
    rate: fitted on the 306 inland talukas at ``quantile=0.98`` it flags 2.3%
    of inland talukas and 28.9% of the held-out Konkan — a 12.6x enrichment on
    a region it never saw.
    """

    def __init__(self, quantile: float = 0.98, shrink: float = 0.3):
        self.quantile = quantile
        self.shrink = shrink
        self.mean_: np.ndarray | None = None
        self.inv_cov_: np.ndarray | None = None
        self.threshold_: float | None = None
        self.train_distances_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray) -> "MahalanobisGuard":
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.mean_ = X.mean(axis=0)
        # more features than samples makes the covariance singular, so it is
        # shrunk toward its diagonal before inversion (Ledoit-Wolf style)
        cov = np.cov(X, rowvar=False)
        cov = (1 - self.shrink) * cov + self.shrink * np.diag(np.diag(cov) + 1e-9)
        self.inv_cov_ = np.linalg.pinv(cov)

        self.train_distances_ = self.distance(X)
        self.threshold_ = float(np.quantile(self.train_distances_, self.quantile))
        return self

    def distance(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        d = X - self.mean_
        return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", d, self.inv_cov_, d), 0, None))

    def is_outlier(self, X) -> np.ndarray:
        return self.distance(X) > self.threshold_

    def novelty(self, X) -> np.ndarray:
        """Distance as a fraction of the threshold — 1.0 is exactly at the cut."""
        return self.distance(X) / (self.threshold_ or 1.0)
