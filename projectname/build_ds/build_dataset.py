from isort import file
import torch
import logging
import gdown
from  hydra.utils import get_original_cwd
import pandas as pd
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from typing import Tuple
import os
from sys import platform

from all_dataclasses.splitting_params import SplittingParams
from all_dataclasses.training_params import TrainParams
from all_dataclasses.pathes_params import PathesParams
from all_dataclasses.predict_params import PredictParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("logs/build_ds.log")
formatter = logging.Formatter("%(asctime)s:%(name)s%(levelname)s:%(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class MyDataset(data.Dataset):
    def __init__(self, csv_path, predict=False):
        self.predict = predict
        self.dataset = pd.read_csv(csv_path)
        if not predict:
            self.x = self.dataset.iloc[:,:-1].to_numpy()
            self.y = self.dataset.iloc[:,-1].to_numpy()
            self.x = torch.from_numpy(self.x)
            self.y = torch.from_numpy(self.y)
        else:
            self.x = self.dataset.iloc[:,:-1].to_numpy()
            self.x = torch.from_numpy(self.x)
            self.y = None


    def __getitem__(self, index):
        if self.predict:
            return self.x[index]
        
        features, target = self.x[index], self.y[index]
        return features, target

    def __len__(self):
      return self.x.shape[0]


def download_data_and_build_predict_dataloader(pred_params: PredictParams) -> DataLoader:
    file_path = f'{get_original_cwd()}/{pred_params.path_to_data}'
    if "win" in platform:
        file_path = '\\'.join(file_path.split('/'))
    if not (os.path.exists(file_path)):
        logger.error("Data not found in {file_path}")
        return None

    logger.info(f"Data found in {file_path}")
    dataset = MyDataset(file_path, predict=True)
    logger.info(f"Predict Dataset len: {len(dataset)}")

    loader = DataLoader(dataset=dataset)
    return loader


def download_data_and_build_dataloaders(train_params: TrainParams,  split_params: SplittingParams, pathes: PathesParams) -> Tuple[DataLoader, DataLoader]:
    
    file_path = f'{get_original_cwd()}/{pathes.input_data_path}'
    if "win" in platform:
        file_path = '\\'.join(file_path.split('/'))
    if not (os.path.exists(file_path)):
        logger.info("downloading data from GoogleDrive")
        url = pathes.data_url
        output = file_path
        gdown.download(url, output)
    logger.info(f"Data can be found in {file_path}")

    dataset = MyDataset(file_path)
    ds_shape = len(dataset)
    logger.info(f"Dataset acquired, Dataset len: {ds_shape}")

    train_size = int(split_params.train_percentage * ds_shape)
    test_size = ds_shape - train_size
    train_dataset, test_dataset = random_split(dataset,(train_size, test_size), generator=torch.Generator().manual_seed(split_params.manual_seed))
    logger.info(f"train_dataset len: {len(train_dataset)}, test_dataset len: {len(test_dataset)}")

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_params.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=train_params.batch_size, shuffle=True)
    return train_loader, test_loader