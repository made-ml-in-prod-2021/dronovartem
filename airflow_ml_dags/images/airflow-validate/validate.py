import os
import click
import pickle
import json

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

TARGET_COL = "target"


def load_model(model_path: str) -> RandomForestClassifier:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def save_metrics(scores: dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(scores, f)


@click.command("validate")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def validate(input_dir: str, model_dir: str, output_dir: str):
    data_path = os.path.join(input_dir, "valid_data.csv")
    data = pd.read_csv(data_path)
    target = data[TARGET_COL]
    data.drop([TARGET_COL], axis=1, inplace=True)

    model_path = os.path.join(model_dir, "model.pkl")
    model = load_model(model_path)

    preds = model.predict(data)
    scores = {"roc_auc": roc_auc_score(target, preds)}

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    save_metrics(scores, metrics_path)


if __name__ == '__main__':
    validate()
