"""Custom metrics module."""

from numpy import log


def get_bic(mse: float, n: int, p: int):
    """
    Calcuate BIC score.

    Parameters
    ----------
    mse: float
        Mean-squared error.
    n: int
        Number of observations.
    p: int
        Number of parameters

    Returns
    -------
    float
        BIC value.
    """
    return n * log(mse) + log(n) * p


def get_aic(mse: float, n: int, p: int):
    """
    Calcuate AIC score.

    Parameters
    ----------
    mse: float
        Mean-squared error.
    n: int
        Number of observations.
    p: int
        Number of parameters

    Returns
    -------
    float
        AIC value.
    """
    return n * log(mse) + 2 * p
