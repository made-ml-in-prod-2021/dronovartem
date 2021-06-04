import os
import click
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TARGET_COL = "target"


def prepare_data(df):
    target = df[TARGET_COL]
    df.drop([TARGET_COL], axis=1, inplace=True)
    return df, target


def train_model(x, y):
    clf = RandomForestClassifier(n_estimators=20)
    clf.fit(x, y)
    return clf


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str):
    data_path = os.path.join(input_dir, "train_data.csv")
    data = pd.read_csv(data_path)
    x, y = prepare_data(data)
    model = train_model(x, y)

    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "model.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()
