import sys
import torch
import hydra
import logging
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from all_dataclasses.train_predict_pipeline_params import set_parametrs
from build_ds.build_dataset import download_data_and_build_dataloaders
from model_code.train_predict_eval import train_model
from model_code.train_predict_eval import serialize_model
from model_code.train_predict_eval import predict
from model_code.train_predict_eval import evaluation
from model_code.train_predict_eval import target_from_loader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("logs/train_pipeline.log")
formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


@hydra.main(config_path="../configs", config_name="config")
def train_pipeline(cfg: DictConfig) -> None:

    train_params, split_params, pathes_params = set_parametrs(cfg)

    logger.info("Started training pipeline with parametrs:")
    logger.info(train_params)
    logger.info(split_params)
    logger.info(pathes_params)

    train_loader, test_loader = download_data_and_build_dataloaders(
        train_params, split_params, pathes_params
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(train_loader, test_loader, train_params, device)

    predict_scores = predict(model, test_loader, device)
    logger.info(f"predict scores size {predict_scores.shape}")

    target = target_from_loader(test_loader)
    logger.info(f"target size {target.shape}")

    metrics = evaluation(predict_scores, target)
    logger.info(f"Evaluated metrics{metrics}")

    file_path = f"{get_original_cwd()}/{pathes_params.output_model_path}"
    if "win" in sys.platform:
        file_path = "\\".join(file_path.split("/"))

    path_to_model = serialize_model(model, file_path)
    return path_to_model, metrics


if __name__ == "__main__":
    train_pipeline()
