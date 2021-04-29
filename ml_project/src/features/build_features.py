import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.entities import FeatureParams
from src.features.custom_transformer import GroupTailTransformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )
    return num_pipeline


def build_numerical_grouped_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("group_tails", GroupTailTransformer()),
            ("scaler", StandardScaler()),
         ]
    )
    return num_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
            (
                "numerical_grouped_pipeline",
                build_numerical_grouped_pipeline(),
                params.numerical_grouped_features,
            ),
        ]
    )
    return transformer
