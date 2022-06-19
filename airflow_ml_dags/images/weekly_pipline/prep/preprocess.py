import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", type=str, default="data.csv", help="input data file path", dest="data_file"
)
parser.add_argument(
    "--processed", type=str, default="train_data.csv", help="train data file path", dest="otput_file"
)
args = parser.parse_args()


x = pd.read_csv(args.data_file + "/data.csv")
y = pd.read_csv(args.data_file + "/target.csv")

x = (x - x.mean()) / x.std()

x['target'] = y['Target'].values

if not os.path.isdir('/'.join(args.otput_file.split("/")[:-1])):
    os.makedirs('/'.join(args.otput_file.split("/")[:-1]))

with open(args.otput_file, "w+") as file:
    file.write(x.to_csv(index=False))
