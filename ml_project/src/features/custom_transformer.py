import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupTailTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.97):
        self.bounds = {}
        self.threshold = threshold

    def fit(self, x: pd.DataFrame, y=None):
        for col in x.columns:
            self.bounds[col] = x[col].quantile(self.threshold)
        return self

    def transform(self, x: pd.DataFrame):
        for col in self.bounds:
            b = self.bounds[col]
            x[col] = x[col].apply(lambda x: b if x > b else x)
        return x.to_numpy()
