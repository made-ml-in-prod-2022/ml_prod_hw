import os
import pickle
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", type=str, default="data.csv", help="input data file path", dest="data_file"
)
parser.add_argument(
    "--predict", type=str, default="predictions.csv", help="predicts file path", dest="otput_file"
)
parser.add_argument(
    "--model", type=str, help="model file path", dest="model_path"
)
args = parser.parse_args()

data = pd.read_csv(args.data_file)

model = pickle.load(open(args.model_path, 'rb'))

pred = model.predict(data)


if not os.path.isdir('/'.join(args.otput_file.split("/")[:-1])):
    os.makedirs('/'.join(args.otput_file.split("/")[:-1]))

with open(args.otput_file, "w+") as file:
    file.write(pd.DataFrame(pred, columns=["prediction"]).to_csv(index=False))
