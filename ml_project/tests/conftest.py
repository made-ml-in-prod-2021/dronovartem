from typing import List

import pytest
import pandas as pd
from faker import Faker


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "thal",
        "ca",
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture
def numerical_grouped_features() -> List[str]:
    return [
        "chol",
    ]


@pytest.fixture()
def tmp_dataset():
    tmp_dataset_size = 10
    faker = Faker()
    Faker.seed(42)
    df = pd.DataFrame({
        "age": [faker.random_int(29, 77) for _ in range(tmp_dataset_size)],
        "sex": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "cp": [0, 1, 3, 1, 0, 2, 0, 1, 3, 1],
        "trestbps": [faker.random_int(90, 200) for _ in range(tmp_dataset_size)],
        "chol": [faker.random_int(120, 560) for _ in range(tmp_dataset_size)],
        "fbs": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "restecg": [2, 1, 0, 1, 2, 1, 0, 1, 0, 1],
        "thalach": [faker.random_int(71, 202) for _ in range(tmp_dataset_size)],
        "exang": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "oldpeak": [faker.random.uniform(0, 6.2) for _ in range(tmp_dataset_size)],
        "slope": [0, 1, 0, 1, 0, 2, 0, 1, 0, 1],
        "ca": [0, 1, 0, 1, 3, 1, 4, 2, 0, 1],
        "thal": [3, 1, 0, 1, 2, 1, 0, 1, 0, 1],
        "target": [faker.random_int(0, 1) for _ in range(tmp_dataset_size)],
    })
    return df
