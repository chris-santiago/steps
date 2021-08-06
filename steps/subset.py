"""Best subsets selection module."""

import itertools
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from steps.metrics import get_aic, get_bic


class SubsetSelector(BaseEstimator, SelectorMixin):
    """Class for best subsets feature selection."""
    def __init__(self, normalize: bool = False, metric: str = 'aic', max_p: Optional[int] = None):
        """
        Constructor method.

        Parameters
        ----------
        normalize: bool
            Whether to normalize data; default = False, assuming object used in pipeline.
        metric: str
            Optimization metric to use; one of ['aic', 'bic'].
        max_p: Optional[int]
            Maximum number of parameters to include; default None.
        """
        self.normalize = normalize
        self.metric = metric
        self.max_p = max_p
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SubsetSelector":
        """
        Fit a best subset regression.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self: object
        """
        X, y = check_X_y(X, y)
        if X.shape[1] > 12:
            raise ValueError("X has too many features for this selection method (max 12).")

        if self.normalize:
            X = self.scaler.fit_transform(X)
        score_funcs = {'aic': get_aic, 'bic': get_bic}
        support = list(itertools.product([True, False], repeat=X.shape[1]))
        if self.max_p:
            support = [s for s in support if sum(s) <= self.max_p]
        n_params = [sum(x) for x in support]
        results = [
            LinearRegression().fit(X[:, mask], y).predict(X[:, mask])
            for mask in support if any(mask)
        ]
        scores = [
            score_funcs[self.metric](mean_squared_error(y, res), len(X), p)
            for res, p in zip(results, n_params)
        ]
        self.best_score_ = min(scores)
        self.best_support_ = np.array(support[scores.index(self.best_score_)])
        return self

    def _get_support_mask(self) -> np.ndarray:
        """
        Get the boolean mask indicating which features are selected

        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """
        check_is_fitted(self)
        return self.best_support_
