from dataclasses import dataclass, field
from typing import Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from .splitting_params import SplittingParams
from .training_params import TrainParams
from .pathes_params import PathesParams
from .predict_params import PredictParams


def set_predict_parametrs(cfg: DictConfig) -> PredictParams:
    return PredictParams(**cfg.predict_params)


def set_parametrs(cfg: DictConfig) -> Tuple[TrainParams, SplittingParams, PathesParams]:
    train_params = TrainParams(**cfg.train_params)
    split_params = SplittingParams(**cfg.splitting_params)
    pathes_params = PathesParams(**cfg.pathes)
    return train_params, split_params, pathes_params


