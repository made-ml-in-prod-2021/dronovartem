import os
import click

import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
def split(input_dir: str):

    data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

    train_data.to_csv(os.path.join(input_dir, "train_data.csv"), index=False)
    valid_data.to_csv(os.path.join(input_dir, "valid_data.csv"),  index=False)


if __name__ == '__main__':
    split()