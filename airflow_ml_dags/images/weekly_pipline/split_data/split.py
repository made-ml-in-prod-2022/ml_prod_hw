import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", type=str, default="data.csv", help="data file path", dest="data_file"
)
parser.add_argument(
    "--split", type=str, help="split data file path", dest="otput_file"
)
args = parser.parse_args()




df = pd.read_csv(args.data_file)
x = df.drop(columns=['target'], axis=1)
y = df['target']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)


if not os.path.isdir(args.otput_file):
    os.makedirs(args.otput_file)

with open(args.otput_file + "/x_train.csv", "w+") as file:
    file.write(x_train.to_csv(index=False))

with open(args.otput_file + "/y_train.csv", "w+") as file:
    file.write(y_train.to_csv(index=False))

with open(args.otput_file + "/x_test.csv", "w+") as file:
    file.write(x_test.to_csv(index=False))

with open(args.otput_file + "/y_test.csv", "w+") as file:
    file.write(y_test.to_csv(index=False))
