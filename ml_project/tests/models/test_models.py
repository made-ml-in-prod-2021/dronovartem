from typing import List

import pytest
from numpy.testing import assert_allclose

from src.entities import (
    FeatureParams, TrainingParams
)
from src.features.build_features import (
    extract_target, build_transformer,
)
from src.models import (
    train_model, evaluate_model,
    predict_model, build_model,
    serialize_model, load_model,
)


@pytest.fixture
def feature_params(
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
        model_type="RandomForestClassifier",
        n_estimators=20,
        random_state=42,
    )
    return params


def test_model_functions(
        tmpdir,
        tmp_dataset,
        tmp_training_params,
        feature_params):
    transformer = build_transformer(feature_params)
    target = extract_target(tmp_dataset, feature_params)

    model = build_model(transformer, train_params=tmp_training_params)
    model = train_model(model, tmp_dataset, target)

    train_pred = predict_model(model, tmp_dataset)

    assert train_pred.shape[0] == target.shape[0]

    model_path = tmpdir.join('model.pkl')
    serialize_model(model, model_path)
    loaded_model = load_model(model_path)
    loaded_model_pred = predict_model(loaded_model, tmp_dataset)
    assert_allclose(train_pred, loaded_model_pred)

    metrics = evaluate_model(train_pred, target)
    assert metrics["roc_auc"] > 0
    assert "log_loss" in metrics.keys()

