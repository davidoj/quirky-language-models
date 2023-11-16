# code from https://github.com/AlignmentResearch/tuned-lens/blob/d512ad05e25c2a67877bb9d042c83cfdfd689aa7/tuned_lens/stats/anomaly.py

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import mahalanobis

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.metrics import RocCurveDisplay


@dataclass
class AnomalyResult:
    """Result of an anomaly detection experiment."""

    model: "BaseEstimator | Mahalanobis"
    """The fitted anomaly detection model."""
    auroc: float
    """The AUROC on the held out mixed data."""
    bootstrapped_aurocs: list[float]
    """AUROCs computed on bootstrapped samples of the held out mixed data."""
    curve: Optional["RocCurveDisplay"]
    """The entire ROC curve on the held out mixed data."""


def bootstrap_auroc(
    labels: np.ndarray, scores: np.ndarray, num_samples: int = 1000, seed: int = 0
) -> list[float]:
    from sklearn.metrics import roc_auc_score

    rng = random.Random(seed)
    n = len(labels)
    aurocs = []

    for _ in range(num_samples):
        idx = rng.choices(range(n), k=n)
        aurocs.append(roc_auc_score(labels[idx], scores[idx]))

    return aurocs


def fit_anomaly_detector(
    normal_x: ArrayLike,
    test_x: ArrayLike,
    test_y: ArrayLike,
    *,
    bootstrap_iters: int = 1000,
    method: Literal["iforest", "lof", "svm", "mahalanobis"] = "mahalanobis",
    plot: bool = True,
    seed: int = 42,
    **kwargs,
) -> AnomalyResult:
    """Fit an unsupervised anomaly detector and test its AUROC on held out mixed data.

    The model only sees normal data during training, but is tested on a mix of normal
    and anomalous data. The AUROC is computed on the held out mixed data.

    Args:
        bootstrap_iters: The number of bootstrap iterations to use for computing the
            95% confidence interval of the AUROC.
        normal: Normal data to train on.
        anomalous: Anomalous data to test on.
        method: The anomaly detection method to use. "iforest" for `IsolationForest`,
            "lof" for `LocalOutlierFactor`, and "svm" for `OneClassSVM`.
        plot: Whether to return a `RocCurveDisplay` object instead of the AUROC.
        seed: The random seed to use for train/test split.
        **kwargs: Additional keyword arguments to pass to the scikit-learn constructor.

    Returns:
        The fitted model, the AUROC, the 95% confidence interval of the AUROC, and the
        entire ROC curve if `plot=True`, evaluated on the held out mixed data.
    """
    # Avoid importing sklearn at module level
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import RocCurveDisplay, roc_auc_score
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    normal_x = np.asarray(normal_x)
    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)
    assert len(normal_x.shape) == 2
    assert normal_x.shape[1] == test_x.shape[1]
    assert len(test_y.shape) == 1
    assert np.unique(test_y).tolist().sort() == [0, 1].sort()

    if method == "iforest":
        model = IsolationForest(**kwargs, random_state=seed).fit(normal_x)
        test_preds = model.score_samples(test_x)
    elif method == "lof":
        model = LocalOutlierFactor(novelty=True, **kwargs).fit(normal_x)
        test_preds = model.decision_function(test_x)
    elif method == "svm":
        model = OneClassSVM(**kwargs).fit(normal_x)
        test_preds = model.decision_function(test_x)
    elif method == "mahalanobis":
        model = Mahalanobis(**kwargs).fit(normal_x)
        test_preds = model.score(test_x)
    else:
        raise ValueError(f"Unknown anomaly detection method '{method}'")

    if plot:
        curve = RocCurveDisplay.from_predictions(test_y, test_preds)
        return AnomalyResult(
            model=model,
            auroc=curve.roc_auc,  # type: ignore
            bootstrapped_aurocs=bootstrap_auroc(test_y, test_preds, bootstrap_iters),
            curve=curve,
        )
    else:
        return AnomalyResult(
            model=model,
            auroc=float(roc_auc_score(test_y, test_preds)),
            bootstrapped_aurocs=bootstrap_auroc(test_y, test_preds, bootstrap_iters),
            curve=None,
        )


class Mahalanobis:
    def __init__(self, subtract_diag_mahal: bool = False):
        self.mean = None
        self.prec = None
        self.subtract_diag_mahal = subtract_diag_mahal

    def fit(self, x: np.ndarray) -> "Mahalanobis":
        self.mean = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])  # numpy returns a scalar for 1D data
        self.prec = np.linalg.inv(cov)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.prec is not None
        dists = mahalanobis(x, self.mean, self.prec) ** 2
        if self.subtract_diag_mahal:
            # a trick Anthropic found to be helpful https://arxiv.org/abs/2204.05862
            dists -= mahalanobis(x, self.mean, np.diag(np.diag(self.prec))) ** 2
        return np.asarray(dists)
