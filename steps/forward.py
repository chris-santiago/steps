"""Forward step selection module."""
from typing import List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from steps.metrics import get_aic, get_bic


class ForwardSelector(BaseEstimator, SelectorMixin):
    """Class for forward stepwise feature selection."""
    def __init__(self, normalize: bool = False, metric: str = 'aic'):
        """
        Constructor method.

        Parameters
        ----------
        normalize: bool
            Whether to normalize data; default = False, assuming object used in pipeline.
        metric: str
            Optimization metric to use; one of ['aic', 'bic'].
        """
        self.normalize = normalize
        self.metric = metric
        self.scaler = StandardScaler()

    @staticmethod
    def _get_null_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        const = np.ones(shape=(len(X), 1))
        return LinearRegression().fit(const, y).predict(const)  # type: ignore

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForwardSelector":
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
        if self.normalize:
            X = self.scaler.fit_transform(X)
        score_func = {'aic': get_aic, 'bic': get_bic}

        best_mse = mean_squared_error(y, self._get_null_fit(X, y))
        best_metric = score_func[self.metric](mse=best_mse, n=len(X), p=0)
        test_metric = 0
        keep_idx: List = []
        params = {i: X[:, i, None] for i in range(X.shape[1])}
        while test_metric <= best_metric:
            if len(keep_idx) == X.shape[1]:  # no more features to test
                break

            # params = [i for i in range(X.shape[1]) if i not in keep_idx]
            if len(keep_idx) == 0:
                results = {
                    i: LinearRegression().fit(x, y).predict(x)
                    for i, x in params.items() if i not in keep_idx
                }
            else:
                results = {
                    i: LinearRegression().fit(X[:, keep_idx+[i]], y).predict(X[:, keep_idx+[i]])
                    for i in params if i not in keep_idx
                }
            scores = {
                i: score_func[self.metric](mean_squared_error(y, res), len(X), len(keep_idx) + 1)
                for i, res in results.items()
            }
            test_metric = min(scores.values())
            if test_metric < best_metric:
                best_metric = test_metric
                keep_idx.append(min(scores, key=scores.get))  # type: ignore
            else:
                break
        # pylint: disable=simplifiable-if-expression
        self.best_support_ = np.array([True if x in keep_idx else False for x in range(X.shape[1])])
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
