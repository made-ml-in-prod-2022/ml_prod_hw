import os
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", type=str, default="data.csv", help="data file path", dest="data_file"
)
parser.add_argument(
    "--target", type=str, default="target.csv", help="target file path", dest="target_file"
)
args = parser.parse_args()


x, y = make_classification(500)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

model = SGDClassifier()

model.fit(x_train, y_train)

pred = model.predict(x_test)

print(accuracy_score(y_test, pred).round(3))


if not os.path.isdir('/'.join(args.data_file.split("/")[:-1])):
    os.makedirs('/'.join(args.data_file.split("/")[:-1]))

with open(args.data_file, "w+") as file:
    file.write(pd.DataFrame(x, columns=[f"Feature_{i}" for i in range(20)]).to_csv(index=False))


if not os.path.isdir('/'.join(args.target_file.split("/")[:-1])):
    os.makedirs('/'.join(args.target_file.split("/")[:-1]))

with open(args.target_file, "w+") as file:
    file.write(pd.DataFrame(y, columns=["Target"]).to_csv(index=False))