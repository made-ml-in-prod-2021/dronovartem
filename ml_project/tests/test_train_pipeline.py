from typing import List

import pytest

from src.train_pipeline import train_pipeline
from src.entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
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


@pytest.fixture()
def tmp_training_params():
    params = TrainingParams(
        model_type="LogisticRegression",
        max_iter=20,
    )
    return params


@pytest.fixture()
def tmp_splitting_params():
    params = SplittingParams(
        val_size=0.2,
        random_state=42,
    )
    return params


def test_train_pipeline(
        tmpdir,
        tmp_dataset,
        tmp_feature_params,
        tmp_training_params,
        tmp_splitting_params):
    input_dataset_path = tmpdir.join("data.csv")
    tmp_dataset.to_csv(input_dataset_path)
    expected_output_model_path = tmpdir.join("model.pkl")

    train_params = TrainingPipelineParams(
        input_data_path=input_dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=tmpdir.join("metrics.json"),
        splitting_params=tmp_splitting_params,
        feature_params=tmp_feature_params,
        train_params=tmp_training_params,
    )
    path, metrics = train_pipeline(train_params)
    assert path == expected_output_model_path
    assert "roc_auc" in metrics.keys()
    assert "log_loss" in metrics.keys()


