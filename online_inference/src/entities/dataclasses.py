from pydantic import BaseModel, conlist, validator
from typing import List, Union

from src.entities import read_features_params

DEFAULT_FEATURES_CONFIG_PATH = 'configs/features_config.yaml'
MODEL_FEATURES = read_features_params(DEFAULT_FEATURES_CONFIG_PATH).features


class HeartDiseaseModelRequest(BaseModel):
    data: List[conlist(Union[float, int])]
    features: List[str]

    @validator('features')
    def validate_model_features(cls, features):
        if not features == MODEL_FEATURES:
            raise ValueError("Wrong setup of features to predict")
        return features


class HeartDiseaseModelResponse(BaseModel):
    class_id: int
