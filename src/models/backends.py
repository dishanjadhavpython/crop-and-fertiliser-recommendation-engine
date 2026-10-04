"""One interface over the model families the tournament compares.

Serving and evaluation both go through these wrappers, so "which model is
better" is answered about the model that would actually serve — the mistake
this project already made once, when the benchmark bagged three seeds and
serving shipped one.

    LGBMBackend      the incumbent: LambdaMART for ranking, quantile for yield
    CatBoostBackend  YetiRank for ranking, MultiQuantile for yield
    TabPFNBackend    TabPFN v2, a tabular foundation model that is fitted in
                     context rather than trained

Three facts about TabPFN shape how it is used here:

* **v2 is pinned deliberately.** Only the v2 weights carry the Prior Labs
  License (Apache 2.0 plus attribution), which permits commercial use with a
  "Built with TabPFN" credit. TabPFN-2.5, 2.6 and 3 are non-commercial, and the
  package defaults to the newest, so the version is selected explicitly.
* **Its context is capped** at roughly 10,000 rows and 500 features. The yield
  task (7,035 rows) fits whole. The ranking grid (30,464 rows) does not, so it
  is subsampled — every planted row kept, the unplanted ones sampled — and
  several such draws are averaged.
* **It has no notion of a ranking objective.** It enters the ranking arm as a
  pointwise model of the relevance grade, which is a real handicap against
  LambdaMART and YetiRank and is reported as such rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config

#: TabPFN v2's documented limits
TABPFN_MAX_ROWS = 10_000
TABPFN_MAX_FEATURES = 500


def available() -> dict[str, str]:
    """Which backends can actually run here, and at what version."""
    out = {"lightgbm": "", "catboost": "", "tabpfn": ""}
    for name in list(out):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "installed")
        except Exception as exc:                      # noqa: BLE001 - report, never raise
            out[name] = f"unavailable: {type(exc).__name__}"
    return out


# ----------------------------------------------------------------- ranking --
class LGBMRankerBackend:
    """The incumbent. Configured by ``ranker.params`` so serving and evaluation agree."""

    name = "lightgbm"

    def __init__(self, seeds: int = 1, **overrides):
        self.seeds, self.overrides, self.models = seeds, overrides, []

    def fit(self, X: pd.DataFrame, y, group) -> "LGBMRankerBackend":
        from lightgbm import LGBMRanker

        from src.models import ranker

        p = ranker.params(list(X.columns), **self.overrides)
        self.models = []
        for s in range(self.seeds):
            m = LGBMRanker(**{**p, "random_state": config.SEED + s})
            m.fit(X, y, group=group)
            self.models.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.models], axis=0)


class CatBoostRankerBackend:
    """YetiRank — a listwise objective, like LambdaMART but a different family.

    CatBoost wants a group id per row rather than group sizes, and it wants the
    rows sorted by that id, which is the same contract LightGBM's ranker has.
    """

    name = "catboost"

    def __init__(self, loss_function: str = "YetiRank", iterations: int = 800,
                 depth: int = 6, learning_rate: float = 0.05, seeds: int = 1,
                 monotone: bool = True, seed: int | None = None, **overrides):
        self.params = dict(loss_function=loss_function, iterations=iterations,
                           depth=depth, learning_rate=learning_rate,
                           verbose=False, allow_writing_files=False, **overrides)
        #: Cross-validation varies this per repeat. Without it every "seed" of a
        #: CatBoost CV refits the identical model and the averaging is a no-op
        #: that merely costs time.
        self.seed = config.SEED if seed is None else seed
        self.seeds, self.monotone, self.models = seeds, monotone, []

    def fit(self, X: pd.DataFrame, y, group) -> "CatBoostRankerBackend":
        from catboost import CatBoostRanker, Pool

        from src.models import ranker

        params = dict(self.params)
        if self.monotone:
            # The same agronomic constraints LightGBM is given. Without them the
            # two arms are not comparable — the incumbent would be carrying a
            # restriction the challenger was free of, and any win could be
            # bought by letting fit quality push a score the wrong way.
            params["monotone_constraints"] = ranker.monotone_constraints(list(X.columns))

        group_id = np.repeat(np.arange(len(group)), group)
        self.models = []
        for s in range(self.seeds):
            pool = Pool(data=X, label=np.asarray(y), group_id=group_id)
            m = CatBoostRanker(**{**params, "random_seed": self.seed + s})
            m.fit(pool)
            self.models.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.models], axis=0)


@dataclass
class TabPFNRankerBackend:
    """TabPFN v2 as a pointwise model of the relevance grade.

    Handicapped by construction — it optimises squared error on a grade, not a
    ranking metric — and capped at ~10k context rows, so each bag keeps every
    planted row and samples the unplanted ones. Both facts belong in the report
    beside its score.
    """

    name = "tabpfn"
    #: two, not four: each bag constructs its own model and pays the weight load,
    #: and the averaging gain is small next to that cost
    bags: int = 2
    max_rows: int = TABPFN_MAX_ROWS
    device: str = "auto"
    models: list = field(default_factory=list)
    columns: list = field(default_factory=list)

    def _regressor(self):
        from tabpfn import TabPFNRegressor
        try:
            from tabpfn.constants import ModelVersion
            return TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        except Exception as exc:                      # noqa: BLE001
            raise RuntimeError(
                "TabPFN v2 weights could not be selected explicitly "
                f"({type(exc).__name__}: {exc}). The package defaults to a newer, "
                "non-commercial version, so the tournament refuses to run rather "
                "than quietly benchmark weights this project may not ship."
            ) from exc

    def fit(self, X: pd.DataFrame, y, group=None) -> "TabPFNRankerBackend":
        y = np.asarray(y, dtype=float)
        if X.shape[1] > TABPFN_MAX_FEATURES:
            raise ValueError(f"TabPFN v2 takes at most {TABPFN_MAX_FEATURES} features; "
                             f"given {X.shape[1]}. Prune before calling.")
        self.columns = list(X.columns)
        rng = np.random.default_rng(config.SEED)
        planted = np.flatnonzero(y > 0)
        rest = np.flatnonzero(y <= 0)

        self.models = []
        for _ in range(self.bags):
            room = max(self.max_rows - len(planted), 0)
            take = rest if len(rest) <= room else rng.choice(rest, room, replace=False)
            idx = np.concatenate([planted, take])[: self.max_rows]
            model = self._regressor()
            model.fit(X.iloc[idx], y[idx])
            self.models.append(model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X[self.columns]) for m in self.models], axis=0)


# ------------------------------------------------------------------- yield --
class LGBMQuantileBackend:
    """The incumbent: one LightGBM per quantile."""

    name = "lightgbm"

    def __init__(self, quantiles=tuple(config.QUANTILES)):
        self.quantiles, self.models = tuple(quantiles), {}

    def fit(self, X: pd.DataFrame, y) -> "LGBMQuantileBackend":
        from lightgbm import LGBMRegressor

        from src.models import yield_quantile

        self.models = {
            a: LGBMRegressor(**yield_quantile.QUANTILE_PARAMS, alpha=a,
                             random_state=config.SEED).fit(X, y)
            for a in self.quantiles
        }
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        return {a: m.predict(X) for a, m in self.models.items()}


class CatBoostQuantileBackend:
    """One CatBoost fitted with MultiQuantile — all three quantiles in one model."""

    name = "catboost"

    def __init__(self, quantiles=tuple(config.QUANTILES), iterations: int = 600,
                 depth: int = 6, learning_rate: float = 0.05):
        self.quantiles = tuple(quantiles)
        self.params = dict(
            loss_function="MultiQuantile:alpha=" + ",".join(str(a) for a in self.quantiles),
            iterations=iterations, depth=depth, learning_rate=learning_rate,
            verbose=False, allow_writing_files=False)
        self.model = None

    def fit(self, X: pd.DataFrame, y) -> "CatBoostQuantileBackend":
        from catboost import CatBoostRegressor

        self.model = CatBoostRegressor(**{**self.params, "random_seed": config.SEED})
        self.model.fit(X, np.asarray(y, dtype=float))
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        pred = np.asarray(self.model.predict(X))
        return {a: pred[:, i] for i, a in enumerate(self.quantiles)}


@dataclass
class TabPFNQuantileBackend:
    """TabPFN v2's own predictive distribution, read at the three quantiles.

    The yield panel is 7,035 rows, inside the context limit, so this arm is not
    subsampled and is the one TabPFN is best suited to.
    """

    name = "tabpfn"
    quantiles: tuple = tuple(config.QUANTILES)
    device: str = "auto"
    model: object = None
    columns: list = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y) -> "TabPFNQuantileBackend":
        if len(X) > TABPFN_MAX_ROWS:
            raise ValueError(f"TabPFN v2 takes at most {TABPFN_MAX_ROWS} context rows; "
                             f"given {len(X)}. Subsample before calling.")
        self.columns = list(X.columns)
        self.model = TabPFNRankerBackend()._regressor()
        self.model.fit(X, np.asarray(y, dtype=float))
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        out = self.model.predict(X[self.columns], output_type="quantiles",
                                 quantiles=list(self.quantiles))
        if isinstance(out, dict):
            return {a: np.asarray(out[a]) for a in self.quantiles}
        return {a: np.asarray(out[i]) for i, a in enumerate(self.quantiles)}


RANKERS = {"lightgbm": LGBMRankerBackend, "catboost": CatBoostRankerBackend,
           "tabpfn": TabPFNRankerBackend}
QUANTILE_MODELS = {"lightgbm": LGBMQuantileBackend, "catboost": CatBoostQuantileBackend,
                   "tabpfn": TabPFNQuantileBackend}
