import sys
import torch
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from  hydra.utils import get_original_cwd

from  all_dataclasses.train_predict_pipeline_params import set_predict_parametrs
from build_ds.build_dataset import download_data_and_build_predict_dataloader
from model_code.train_predict_eval import predict
from model_code.train_predict_eval import load_model
from model_code.train_predict_eval import save_results


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("logs/predict_pipeline.log")
formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


@hydra.main(config_path='../configs', config_name="config")
def predict_pipeline(cfg: DictConfig) -> None:

    predict_params = set_predict_parametrs(cfg)

    logger.info("Started predict pipeline with parametrs:")
    logger.info(predict_params)

    pred_loader = download_data_and_build_predict_dataloader(predict_params)
    if pred_loader is None:
        logger.error("Pipeline stopped, no data found")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(predict_params.path_to_model)

    predict_scores = predict(model, pred_loader, device)
    logger.info(f"predict scores size {predict_scores.shape}")

    file_path = f'{get_original_cwd()}/{predict_params.output_path}'
    if "win" in sys.platform:
        file_path = '\\'.join(file_path.split('/'))
    
    res_path = save_results(predict_scores, file_path)
    return res_path


if __name__ == "__main__":
    predict_pipeline()