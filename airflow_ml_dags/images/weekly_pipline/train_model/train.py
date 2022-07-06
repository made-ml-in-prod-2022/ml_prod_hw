import os
import pandas as pd
import argparse
import pickle
from sklearn.linear_model import SGDClassifier


parser = argparse.ArgumentParser()
parser.add_argument(
    "--split", type=str, help="split data dir path", dest="input_dir"
)
parser.add_argument(
    "--model", type=str, help="model file path", dest="model_path"
)
args = parser.parse_args()


x_train = pd.read_csv(args.input_dir + "/x_train.csv")
y_train = pd.read_csv(args.input_dir + "/y_train.csv")

model = SGDClassifier()

model.fit(x_train, y_train)

if not os.path.isdir(args.model_path):
    os.makedirs(args.model_path)

pickle.dump(model, open(args.model_path + "/model.pkl", 'wb'))
