import json
import sys
import torch
import numpy as np
import pickle
import logging
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from all_dataclasses.training_params import TrainParams

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("logs/train.log")
formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.WARNING)


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


def check_accuracy(loader: DataLoader, model: NN, device: int) -> float:
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x.float())
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


def predict(model: NN, loader: DataLoader, device: int) -> np.array:
    model.eval()
    res_scores = np.array([])
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            scores = model(x.float())
            _, predictions = scores.max(1)
            res_scores = np.concatenate((res_scores, predictions.cpu()))
    return res_scores


def target_from_loader(loader: DataLoader) -> np.array:
    result = np.empty(len(loader))
    for i, (val, target) in enumerate(loader):
        result[i] = target
    return result


def evaluation(predict: np.array, ground_truth: np.array) -> json:
    return {"accuracy": np.round(accuracy_score(ground_truth, predict), 3)}


def serialize_model(model: torch.nn, output: str) -> str:
    logger.info("Model serialized, put in {output}")
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def train_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_params: TrainParams,
    device: int,
) -> NN:
    torch.manual_seed(train_params.SEED)
    model = NN(
        input_size=train_params.input_size, num_classes=train_params.num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_params.learning_rate,
        weight_decay=train_params.weight_decay,
    )
    # optimizer = optim.SGD(model.parameters(), nesterov=True, momentum=0.9, lr=0.1, weight_decay=0.0001)

    train_acc = []
    test_acc = []

    for epoch in range(train_params.num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data.float())
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        current_acc_train = check_accuracy(train_loader, model, device).cpu()
        current_acc_test = check_accuracy(test_loader, model, device).cpu()
        train_acc.append(current_acc_train)
        test_acc.append(current_acc_test)

        logger.info(f"Epoch: {epoch}: Accuracy on training set: {current_acc_train}")
        logger.info(f"Epoch: {epoch}: Accuracy on test set: {current_acc_test}")

    return model


def load_model(path: str) -> NN:
    logger.info(f"Model loaded, from {path}")
    with open(path, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


def save_results(scores: np.array, path: str) -> str:
    logger.info(f"Resuls saved in {path}")
    with open(path, "w+") as f:
        f.write(scores)
    return path
