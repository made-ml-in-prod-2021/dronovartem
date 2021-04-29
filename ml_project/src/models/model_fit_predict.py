import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from src.entities.train_params import TrainingParams


def build_model(transformer: TransformerMixin, train_params: TrainingParams
                ) -> Pipeline:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators,
            random_state=train_params.random_state,
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=train_params.max_iter)
    else:
        raise NotImplementedError()

    clf_model = Pipeline(
        steps=[('preprocessor', transformer),
               ('classifier', model),
               ])
    return clf_model


def train_model(
        model: Pipeline,
        features: pd.DataFrame, target: pd.Series,
) -> Pipeline:
    model.fit(features, target)
    return model


def predict_model(
        model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
        predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "roc_auc": roc_auc_score(target, predicts),
        "log_loss": log_loss(target, predicts),
    }


def serialize_model(model: Pipeline, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(model_path: str) -> Pipeline:
    with open(model_path, "rb") as mp:
        model = pickle.load(mp)
    return model
