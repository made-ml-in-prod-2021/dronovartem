import pytest
import numpy as np
import pandas as pd

from src.features.custom_transformer import (
    GroupTailTransformer
)

SMALL_DATASET_SIZE = 10


@pytest.fixture()
def small_df():
    df = pd.DataFrame({
        "feat1": [i for i in range(SMALL_DATASET_SIZE)],
        "feat2": [i ** 2 for i in range(SMALL_DATASET_SIZE)]
    })
    return df


def test_custom_transformer_fit(small_df):
    gt = GroupTailTransformer(threshold=.9)
    gt.fit(small_df)
    assert len(gt.bounds) == 2


def test_custom_transformer_transform(small_df):
    gt = GroupTailTransformer(threshold=.9)
    gt.fit(small_df)
    out = gt.transform(small_df)
    assert (np.max(out[:, 0]) - 8.1) < 1e-6
    assert (np.min(out[:, 0]) - 0) < 1e-6
    assert (np.max(out[:, 1]) - 65.7) < 1e-6
