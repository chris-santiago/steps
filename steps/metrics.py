"""Custom metrics module."""

from numpy import log


def get_bic(loss: float, n: int, p: int):
    """
    Calcuate BIC score.

    Parameters
    ----------
    loss: float
        Model loss (MSE or Log-Loss).
    n: int
        Number of observations.
    p: int
        Number of parameters

    Returns
    -------
    float
        BIC value.
    """
    return n * log(loss) + log(n) * p


def get_aic(loss: float, n: int, p: int):
    """
    Calcuate AIC score.

    Parameters
    ----------
    loss: float
        Model loss (MSE or Log-Loss).
    n: int
        Number of observations.
    p: int
        Number of parameters

    Returns
    -------
    float
        AIC value.
    """
    return n * log(loss) + 2 * p
