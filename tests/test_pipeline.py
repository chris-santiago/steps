import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from steps.forward import ForwardSelector
from steps.subset import SubsetSelector


class TestForwardSelector:
    @pytest.mark.parametrize(
        'metric, n_params',
        [('aic', 4), ('bic', 4)]  # ! Note: AIC will typically be +1 BIC in smaller sample sizes.
    )
    def test_selected_params(self, metric, n_params, dummy_train_test):
        X_train, X_test, y_train, y_test = dummy_train_test
        pl = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', ForwardSelector(metric=metric)),
            ('lr', LinearRegression())
        ])
        pl.fit(X_train, y_train)
        expected_params = n_params
        actual_params = pl['selector'].get_support().sum()
        assert actual_params == expected_params

    @pytest.mark.parametrize(
        'metric, n_params',
        [('aic', 4), ('bic', 4)]
    )
    def test_features_in(self, metric, n_params, dummy_train_test):
        X_train, X_test, y_train, y_test = dummy_train_test
        pl = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', ForwardSelector(metric=metric)),
            ('lr', LinearRegression())
        ])
        pl.fit(X_train, y_train)
        expected_params = n_params
        actual_lr_features = pl['lr'].n_features_in_
        assert expected_params == actual_lr_features


class TestSubsetSelector:
    @pytest.mark.parametrize(
        'metric, n_params',
        [('aic', 4), ('bic', 4)]
    )
    def test_selected_params(self, metric, n_params, dummy_train_test):
        X_train, X_test, y_train, y_test = dummy_train_test
        pl = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SubsetSelector(metric=metric)),
            ('lr', LinearRegression())
        ])
        pl.fit(X_train, y_train)
        expected_params = n_params
        actual_params = pl['selector'].get_support().sum()
        assert actual_params == expected_params

    @pytest.mark.parametrize(
        'metric, n_params',
        [('aic', 4), ('bic', 4)]
    )
    def test_features_in(self, metric, n_params, dummy_train_test):
        X_train, X_test, y_train, y_test = dummy_train_test
        pl = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SubsetSelector(metric=metric)),
            ('lr', LinearRegression())
        ])
        pl.fit(X_train, y_train)
        expected_params = n_params
        actual_lr_features = pl['lr'].n_features_in_
        assert actual_lr_features == expected_params
