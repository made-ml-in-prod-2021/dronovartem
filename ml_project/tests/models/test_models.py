from typing import List

import pytest
from sklearn.ensemble import RandomForestClassifier

from src.entities import (
    FeatureParams, TrainingParams
)
from src.features.build_features import (
    make_features, extract_target, build_transformer,
)
from src.models import (
    train_model, evaluate_model, predict_model,
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


def test_model_train_predict_evaluate(
        tmp_dataset,
        tmp_training_params,
        feature_params):
    transformer = build_transformer(feature_params)
    transformer.fit(tmp_dataset)
    features = make_features(transformer, tmp_dataset)
    target = extract_target(tmp_dataset, feature_params)

    model = train_model(features, target, train_params=tmp_training_params)
    isinstance(model, RandomForestClassifier)

    train_pred = predict_model(model, features)
    assert train_pred.shape[0] == target.shape[0]

    metrics = evaluate_model(train_pred, target)
    assert metrics["roc_auc"] > 0
    assert "log_loss" in metrics.keys()

