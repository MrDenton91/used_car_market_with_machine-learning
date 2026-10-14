"""The price model: LightGBM with native categorical handling.

Kept in its own module so a saved model can be unpickled by predict.py (a class
defined in train.py would be pickled as __main__.CategoricalLGBM).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split


class CategoricalLGBM(BaseEstimator, RegressorMixin):
    """LightGBM regressor that learns the category levels of the string columns
    at fit time (unseen levels at predict time become missing), holds out 10%
    of the rows for early stopping, and optionally fits log(price)."""

    def __init__(self, categorical, target='raw', n_estimators=4000, learning_rate=0.05, num_leaves=127,
                 min_child_samples=40, reg_lambda=1.0, random_state=10):
        self.categorical = categorical
        self.target = target
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def _encode(self, X):
        X = X.copy()
        for col in self.categorical:
            X[col] = pd.Categorical(X[col].astype(str), categories=self.categories_[col])
        return X

    def fit(self, X, y):
        import lightgbm as lgb
        self.categories_ = {col: pd.Index(X[col].astype(str).unique()) for col in self.categorical}
        self.columns_ = list(X.columns)
        X = self._encode(X)
        y = np.log(y) if self.target == 'log' else np.asarray(y, dtype=float)
        X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.1, random_state=self.random_state)
        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate, num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples, reg_lambda=self.reg_lambda,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=20,
            verbose=-1, random_state=self.random_state)
        self.model_.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        self.n_trees_ = self.model_.best_iteration_
        return self

    def predict(self, X):
        pred = self.model_.predict(self._encode(X[self.columns_]))
        return np.exp(pred) if self.target == 'log' else pred
