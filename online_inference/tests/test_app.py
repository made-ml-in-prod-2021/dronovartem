from fastapi.testclient import TestClient

import pytest
import pandas as pd

from src.app import app

DEFAULT_SUCCESS_STATUS_CODE = 200
DEFAULT_VALIDATION_ERROR_CODE = 400


def test_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == DEFAULT_SUCCESS_STATUS_CODE


@pytest.fixture(scope='session')
def tmp_scored_data():
    features = ['age', 'sex', 'cp',
                'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang',
                'oldpeak', 'slope', 'ca', 'thal']
    data = [0] * len(features)
    return pd.DataFrame({feature: [value] for feature, value in zip(features, data)})


@pytest.fixture(scope='session')
def model_path():
    return 'models/model.pkl'


def test_predict_request_validation_good_data(model_path, tmp_scored_data):
    with TestClient(app) as client:
        data = tmp_scored_data.values.tolist()
        features = tmp_scored_data.columns.tolist()
        response = client.get("/predict/",
                              json={"data": data, "features": features},
                              )
        assert response.status_code == DEFAULT_SUCCESS_STATUS_CODE


def test_predict_request_validation_lack_of_data(model_path, tmp_scored_data):
    with TestClient(app) as client:
        data = tmp_scored_data.values.tolist()
        features = tmp_scored_data.columns.tolist()
        response = client.get("/predict/",
                              json={"data": data[1:], "features": features[1:]},
                              )
        assert response.status_code == DEFAULT_VALIDATION_ERROR_CODE


def test_predict_request_validation_excess_of_data(model_path, tmp_scored_data):
    tmp_scored_data = tmp_scored_data.assign(extra_column=lambda x: x.age)
    with TestClient(app) as client:
        data = tmp_scored_data.values.tolist()
        features = tmp_scored_data.columns.tolist()
        response = client.get("/predict/",
                              json={"data": data, "features": features},
                              )
        assert response.status_code == DEFAULT_VALIDATION_ERROR_CODE


def test_predict_request_validation_permute_of_data(model_path, tmp_scored_data):
    columns = tmp_scored_data.columns.tolist()
    tmp_scored_data = tmp_scored_data[columns[1:] + columns[:1]]
    with TestClient(app) as client:
        data = tmp_scored_data.values.tolist()
        features = tmp_scored_data.columns.tolist()
        response = client.get("/predict/",
                              json={"data": data, "features": features},
                              )
        assert response.status_code == DEFAULT_VALIDATION_ERROR_CODE
