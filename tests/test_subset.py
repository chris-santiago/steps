from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import NotFittedError

from steps.subset import SubsetSelector


class TestSubsetSelector:
    def test_max_features_raises(self, dummy_data_high_p):
        X, y, coef = dummy_data_high_p
        with pytest.raises(ValueError):
            selector = SubsetSelector()
            final = selector.fit_transform(X, y)

    def test_fit_aic(self, dummy_data):
        X, y, coef = dummy_data
        selector = SubsetSelector()
        final = selector.fit_transform(X, y)
        np.testing.assert_equal(coef*selector.best_support_, coef)

    def test_fit_bic(self, dummy_data):
        X, y, coef = dummy_data
        selector = SubsetSelector(metric='bic')
        final = selector.fit_transform(X, y)
        np.testing.assert_equal(coef*selector.best_support_, coef)

    def test_not_fitted(self):
        selector = SubsetSelector()
        with pytest.raises(NotFittedError):
            selector.get_support()

    def test_normalized(self, dummy_data, monkeypatch):
        X, y, coef = dummy_data
        selector = SubsetSelector(normalize=True)
        mock_scaler = Mock()
        mock_scaler.fit_transform.return_value = X
        monkeypatch.setattr(selector, 'scaler', mock_scaler)
        final = selector.fit_transform(X, y)
        mock_scaler.fit_transform.assert_called_once()

    def test_not_normalized(self, dummy_data, monkeypatch):
        X, y, coef = dummy_data
        selector = SubsetSelector(normalize=False)
        mock_scaler = Mock()
        mock_scaler.fit_transform.return_value = X
        monkeypatch.setattr(selector, 'scaler', mock_scaler)
        final = selector.fit_transform(X, y)
        mock_scaler.fit_transform.assert_not_called()

    def test_pandas(self, dummy_data):
        X, y, coef = dummy_data
        X = pd.DataFrame(X)
        y = pd.Series(y)
        selector = SubsetSelector()
        final = selector.fit_transform(X, y)
        np.testing.assert_equal(coef * selector.best_support_, coef)
