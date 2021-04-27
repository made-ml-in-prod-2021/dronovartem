from typing import List
import pytest
import pandas as pd
from numpy.testing import assert_allclose

from src.entities.feature_params import FeatureParams
from src.features.build_features import (
    make_features, extract_target, build_transformer,
    build_categorical_pipeline, build_numerical_pipeline,
    build_numerical_grouped_pipeline,
)


@pytest.fixture
def tmp_feature_params(
        categorical_features: List[str],
        numerical_features: List[str],
        numerical_grouped_features: List[str],
        target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        numerical_grouped_features=numerical_grouped_features,
        target_col=target_col,
    )
    return params


def test_make_features(
        tmp_feature_params: FeatureParams, tmp_dataset,
):
    transformer = build_transformer(tmp_feature_params)
    transformer.fit(tmp_dataset)
    features = make_features(transformer, tmp_dataset)
    assert features.shape[1] == 30, (
        f"Its expected to have 30 features after transformer, got {features.shape[1]}"
    )


def test_extract_target(
        tmp_feature_params: FeatureParams, tmp_dataset,
):
    extracted_target = extract_target(tmp_dataset, tmp_feature_params)
    assert_allclose(extracted_target, tmp_dataset[tmp_feature_params.target_col])


@pytest.fixture()
def tmp_cat_dataset():
    return pd.DataFrame({"pos": ["first", "second", "first"]})


def test_build_categorical_pipeline(
        tmp_cat_dataset,
):
    pipe_output = build_categorical_pipeline().fit_transform(tmp_cat_dataset)
    assert pipe_output.shape == (3, 2)
    assert pipe_output[:, 0].sum() == 2


@pytest.fixture()
def tmp_num_dataset():
    return pd.DataFrame({"value": [i for i in range(10)]})


def test_build_numerical_pipeline(
        tmp_num_dataset,
):
    pipe_output = build_numerical_pipeline().fit_transform(tmp_num_dataset)
    assert pipe_output.shape == (10, 1)
    assert abs(pipe_output[:, 0].mean() - 0) < 1e-6
    assert abs(pipe_output[:, 0].std() - 1) < 1e-6


def test_build_numerical_grouped_pipeline(
        tmp_num_dataset,
):
    pipe_output = build_numerical_grouped_pipeline().fit_transform(tmp_num_dataset)
    assert pipe_output.shape == (10, 1)
    assert abs(pipe_output[:, 0].mean() - 0) < 1e-6
    assert abs(pipe_output[:, 0].std() - 1) < 1e-6
