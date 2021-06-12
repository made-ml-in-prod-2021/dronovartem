import os
import click
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def load_model(model_path: str) -> RandomForestClassifier:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def save_predictions(preds, output_path: str):
    y = pd.DataFrame({"target": preds})
    y.to_csv(output_path, index=False, header=None)


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    data_path = os.path.join(input_dir, "data.csv")
    data = pd.read_csv(data_path)

    model_path = os.path.join(model_dir, "model.pkl")
    model = load_model(model_path)

    preds = model.predict(data)
    os.makedirs(output_dir, exist_ok=True)
    preds_path = os.path.join(output_dir, "predictions.csv")
    save_predictions(preds, preds_path)


if __name__ == '__main__':
    predict()
