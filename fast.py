import json
import logging
import os
import sys
import gdown
from typing import List
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np
import pandas as pd
import uvicorn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def load_model_pkl(path: str) -> NN:
    logger.info(f"Model loaded, from {path}")
    loaded_model = NN(input_size=13, num_classes=2)
    loaded_model.load_state_dict(torch.load("model/model.pkl"))
    loaded_model.eval()
    return loaded_model


class HeartDecease(BaseModel):
    data: List[List[float]]
    features: List[str]

    @validator("features")
    def fetures_check(cls, features):
        correct_features = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
        if len(features) != len(correct_features):
            raise HTTPException(status_code=400, detail="features len is not right")

        for i, feature in enumerate(correct_features):
            if features[i] != feature:
                raise HTTPException(
                    status_code=400, detail="features order is not right"
                )
        return features

    @validator("data")
    def data_check(cls, data):
        for row in data:
            if len(row) != 13:
                raise HTTPException(status_code=400, detail="data len is not 13")

            for i, value in enumerate(row):
                if not isinstance(value, float):
                    raise HTTPException(
                        status_code=400, detail="data is not in correct type"
                    )
        return data


class Response(BaseModel):
    condition: bool


model: NN = None


def predict_nn(loader: DataLoader) -> np.array:
    logger.info("in predict func")
    model.eval()
    res_scores = np.array([])
    with torch.no_grad():
        for x in loader:
            # x = x.reshape(x.shape[0], -1)
            logger.info(x.shape)
            scores = model(x.float())
            _, predictions = scores.max(1)
            res_scores = np.concatenate((res_scores, predictions))
    return res_scores


class MyDataset(data.Dataset):
    def __init__(self, dataset, predict=False):
        self.predict = predict
        self.dataset = dataset
        self.x = self.dataset.iloc[:, :].to_numpy()
        self.x = torch.from_numpy(self.x)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]


def make_predict(outu_data: List, features: List[str]) -> List[Response]:
    data = pd.DataFrame(outu_data, columns=features)

    logger.info(data)

    dataset = MyDataset(data)

    logger.info(f"Predict Dataset len: {len(dataset)}")

    loader = DataLoader(dataset=dataset)

    predicts = predict_nn(loader)

    logger.info(f"predicts: {predicts}")

    return [Response(condition=bool(cond)) for id, cond in enumerate(predicts)]


app = FastAPI()


@app.get("/")
def main():
    return {"msg": "Check /predict or /health"}


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")

    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    if not (os.path.exists(model_path)):
        logger.info("downloading data from GoogleDrive")
        url = "https://drive.google.com/u/0/uc?id=1ECaTFW1WvAeUDiD560fKEARc3qk85pGK&export=download"
        output = model_path
        gdown.download(url, output)

    model = load_model_pkl(model_path)


@app.get("/health", status_code=200)
def health() -> json:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return {"model": "ready"}


@app.get("/predict/", response_model=List[Response])
def predict(request: HeartDecease):
    return make_predict(request.data, request.features)


if __name__ == "__main__":
    uvicorn.run("fast:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
