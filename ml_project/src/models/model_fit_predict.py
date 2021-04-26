import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from src.entities.train_params import TrainingParams

ClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
        features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> ClassificationModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators,
            random_state=train_params.random_state,
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=train_params.max_iter)
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
        model: ClassificationModel, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
        predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "roc_auc": roc_auc_score(target, predicts),
        "log_loss": log_loss(target, predicts),
    }


def serialize_model(model: ClassificationModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
