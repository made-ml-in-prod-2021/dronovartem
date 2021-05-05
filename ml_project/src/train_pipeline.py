import sys
import logging
import json

from argparse import ArgumentParser

from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from src.data import (
    read_data,
    split_train_val_data,
)
from src.features.build_features import (
    build_transformer,
    extract_target,
)
from src.models import (
    build_model,
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DEFAULT_CONFIG_PATH = './configs/train_rf_config.yaml'


def setup_parser(parser):
    """
    Add arguments to given parser.
    :param parser: argparse.ArgumentParser
    """
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        dest="config_path",
        help="path to config yaml", )


def setup_pipeline_params(config_path: str):
    """
    Extract params from given yaml config path.
    :param config_path: Path to .yaml with config.
    :return: training pipeline params
    """
    params = read_training_pipeline_params(config_path)
    return params


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    """
    Include base logic of pipeline training:
    read train data, create validation data, train and valid model, save model and score.
    :param training_pipeline_params: training pipeline params
    :return: tuple with path to serialized model and its metrics.
    """
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    dataset = read_data(training_pipeline_params.input_data_path)
    logger.info(f"loaded dataset from {training_pipeline_params.input_data_path}")
    train_df, valid_df = split_train_val_data(dataset, training_pipeline_params.splitting_params)

    transformer = build_transformer(training_pipeline_params.feature_params)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    logger.info(f"Train target shape is {train_target.shape}")

    logger.info("Begin train model...")
    clf_model = build_model(transformer, train_params=training_pipeline_params.train_params)
    model = train_model(clf_model, train_df, train_target)
    logger.info("Model training has finished. ")

    logger.info("Prepare to model validation... ")
    val_target = extract_target(valid_df, training_pipeline_params.feature_params)

    predicts = predict_model(
        model,
        valid_df,
    )
    logger.info(f"Compute validation scores...")
    metrics = evaluate_model(
        predicts,
        val_target,
    )

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"Metrics are {metrics}")

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)
    logger.info("Model was serialized and saved. ")
    return path_to_model, metrics


def train_pipeline_command():
    """
    Include CLI logic and run training.
    """
    parser = ArgumentParser()
    setup_parser(parser)
    arguments = parser.parse_args()
    params = setup_pipeline_params(arguments.config_path)
    train_pipeline(params)


if __name__ == '__main__':
    train_pipeline_command()
