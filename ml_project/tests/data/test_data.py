import pytest

import pandas as pd
from faker import Faker

from src.data import (
    read_data,
    split_train_val_data,
)
from src.entities import SplittingParams

SMALL_DATASET_SIZE = 10


@pytest.fixture()
def small_dataset(tmpdir):
    dataset_fio = tmpdir.join("dataset.csv")
    faker = Faker()
    dataset = pd.DataFrame({"x": [faker.random_int(0, 5) for _ in range(SMALL_DATASET_SIZE)]})
    dataset.to_csv(dataset_fio)
    return dataset_fio


def test_make_dataset(small_dataset):
    df = read_data(small_dataset)
    assert df.shape[0] == SMALL_DATASET_SIZE


def test_split_train_val_data(small_dataset):
    val_size = 0.2
    data = read_data(small_dataset)
    params = SplittingParams(val_size=val_size, random_state=42)
    train_df, valid_df = split_train_val_data(data, params)
    assert train_df.shape[0] == (1 - val_size) * SMALL_DATASET_SIZE
    assert valid_df.shape[0] == val_size * SMALL_DATASET_SIZE
