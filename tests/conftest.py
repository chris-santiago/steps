import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

N = 400
NOISE = 1


@pytest.fixture
def dummy_data():
    return make_regression(
        n_samples=N, n_features=10, n_informative=4, bias=46, noise=NOISE, coef=True, random_state=42
    )


@pytest.fixture
def dummy_data_high_p():
    return make_regression(
        n_samples=N, n_features=13, n_informative=4, bias=46, noise=NOISE, coef=True, random_state=42
    )


@pytest.fixture
def dummy_data_all_relevant():
    return make_regression(
        n_samples=N, n_features=10, n_informative=10, bias=46, noise=NOISE, coef=True, random_state=42
    )


@pytest.fixture
def dummy_data_one_relevant():
    return make_regression(
        n_samples=N, n_features=1, n_informative=10, bias=46, noise=NOISE, coef=True, random_state=42
    )


@pytest.fixture
def dummy_train_test():
    X, y, coef = make_regression(
        n_samples=N, n_features=10, n_informative=4, bias=46, noise=NOISE, coef=True, random_state=42
    )
    return train_test_split(X, y, test_size=.2, random_state=42)
