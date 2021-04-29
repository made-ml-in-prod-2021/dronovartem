import sys
import logging

from argparse import ArgumentParser

from src.entities.predict_pipeline_params import (
    PredictPipelineParams,
    read_predict_pipeline_params,
)
from src.data import (
    read_data,
    save_data,
    target_to_dataframe,
)
from src.models import (
    load_model,
    predict_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DEFAULT_CONFIG_PATH = './configs/predict_config.yaml'


def setup_parser(parser, default_params: PredictPipelineParams):
    """
    Add arguments to given parser.
    :param default_params: Used to get
    :param parser: argparse.ArgumentParser
    """
    parser.add_argument(
        "--data",
        default=default_params.input_data_path,
        dest="input_data_path",
        help="path to data to use for prediction", )

    parser.add_argument(
        "--model",
        default=default_params.model_path,
        dest="model_path",
        help="path to model", )

    parser.add_argument(
        "--output",
        default=default_params.output_path,
        dest="output_path",
        help="path to model", )


def setup_pipeline_params(config_path: str):
    """
    Extract params from given yaml config path.
    :param config_path: Path to .yaml with config.
    :return: 0 as a success code
    """
    params = read_predict_pipeline_params(config_path)
    return params


def predict_pipeline(arguments):
    """
    Perform a prediction based on CLI params.
    :param arguments: Consist of input_path, model_path and
    :return:
    """
    logger.info(f"start predict pipeline with params {arguments}")
    dataset = read_data(arguments.input_data_path)
    logger.info(f"loaded dataset from {arguments.input_data_path}")
    logger.info(f"Examples count is {dataset.shape[0]}")

    logger.info(f"Load model from {arguments.model_path}")
    model = load_model(arguments.model_path)
    logger.info("Model loading has finished. ")

    logger.info("Start making predictions...")
    predicts = predict_model(
        model,
        dataset,
    )
    logger.info("Predictions computed. Save to file. ")
    save_data(target_to_dataframe(predicts), arguments.output_path)
    logger.info(f"Predictions successfully saved in {arguments.output_path}")
    return 0


def predict_pipeline_command():
    """
    Include CLI logic and run training.
    """
    parser = ArgumentParser()
    default_params = setup_pipeline_params(DEFAULT_CONFIG_PATH)
    setup_parser(parser, default_params)
    arguments = parser.parse_args()
    predict_pipeline(arguments)


if __name__ == '__main__':
    predict_pipeline_command()
