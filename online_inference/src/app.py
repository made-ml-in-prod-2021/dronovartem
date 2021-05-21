import logging
import sys
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from sklearn.pipeline import Pipeline

from src.entities import (
    read_app_params,
    HeartDiseaseModelRequest, HeartDiseaseModelResponse,
)
from src.models import make_predict, load_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DEFAULT_CONFIG_PATH = "configs/app_config.yaml"
model: Optional[Pipeline] = None
app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_app_model():
    app_params = read_app_params("configs/app_config.yaml")
    logger.info("Start loading model")
    global model
    model = load_model(app_params.model_path)
    logger.info("Model loaded")


@app.get("/predict/", response_model=List[HeartDiseaseModelResponse])
def predict(request: HeartDiseaseModelRequest):
    return make_predict(request.data, request.features, model)


def setup_app():
    app_params = read_app_params(DEFAULT_CONFIG_PATH)
    logger.info(f"Running app on {app_params.host} with port {app_params.port}")
    uvicorn.run(app, host=app_params.host, port=app_params.port)


if __name__ == "__main__":
    setup_app()
