"""Step selection mixin module."""
from abc import ABCMeta
from typing import Any, Union

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error


class StepsMixin(metaclass=ABCMeta):
    """
    Step selection mixin that returns regressor/classifier estimator and score func.

    This mixin provides an estimator based on target dtype using the `get_estimator` method
    and a score func based on target dtype using the `get_loss_func` func.
    """

    @staticmethod
    def get_estimator(y: np.ndarray) -> Any:
        """
        Get an estimator for subset/stepwise feature selection.

        Parameters
        ----------
        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        Type[LinearRegression, LogisticRegression]
            A Scikit-learn estimator.
        """
        if y.dtype == 'float':
            return LinearRegression
        return LogisticRegression

    @staticmethod
    def get_loss_func(y: np.ndarray) -> Union[mean_squared_error, log_loss]:
        """
        Get a loss function for subset/stepwise feature selection.

        Parameters
        ----------
        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        Union[mean_squared_error, log_loss]
            A Scikit-learn loss function.
        """
        if y.dtype == 'float':
            return mean_squared_error
        return log_loss
