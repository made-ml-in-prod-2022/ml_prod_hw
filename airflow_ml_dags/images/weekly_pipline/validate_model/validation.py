import os
import pickle
import json
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split", type=str, help="split data dir path", dest="input_dir"
)
parser.add_argument(
    "--model", type=str, help="model file path", dest="model_path"
)
args = parser.parse_args()


x_test = pd.read_csv(args.input_dir + "/x_test.csv")
y_test = pd.read_csv(args.input_dir + "/y_test.csv")

model = pickle.load(open(args.model_path + "/model.pkl", 'rb'))

pred = model.predict(x_test)

accuracy = accuracy_score(y_test, pred).round(3)
metrics = {
    "accurracy": accuracy,
    "precision": "-",
    "recall": "-"
}


if not os.path.isdir(args.model_path):
    os.makedirs(args.model_path)

with open(args.model_path + "/metrics.json", "w+") as file:
    file.write(json.dumps(metrics))
