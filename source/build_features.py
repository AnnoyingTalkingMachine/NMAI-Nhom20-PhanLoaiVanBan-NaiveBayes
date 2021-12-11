# Đọc dữ liệu raw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataFrame_raw = pd.read_csv("C:\\Users\\Long\\Programming\\NMAI-Nhom20-PhanLoaiVanBan-NaiveBayes\\data\\raw\\demodataset.csv", encoding="ISO-8859-1", header=None)
dataFrame_raw.columns = ["label", "time", "date", "query", "username", "text"]

dataFrame = dataFrame_raw[["label", "text"]]


# Cắt nhỏ kích thước dữ liệu
dataFrame_positive = dataFrame[dataFrame["label"] == 4]
dataFrame_negative = dataFrame[dataFrame["label"] == 0]

# dataFrame_positive = dataFrame_positive.iloc[:int(len(dataFrame_positive) / 40)]
# dataFrame_negative = dataFrame_negative.iloc[:int(len(dataFrame_negative) / 40)]

dataFrame = pd.concat([dataFrame_positive, dataFrame_negative])

# [ (string, label) ] -> [ ( {token: số lần lặp}, label ) ]
from tokenize_clean_data import tokenize_clean_sentence

data = []
for index, df in dataFrame.iterrows():
    if df["label"] == 4:
        data.append( (tokenize_clean_sentence(df["text"]), 1) )
    else:
        data.append( (tokenize_clean_sentence(df["text"]), 0) )

# Chia thành tập train và test
import random

random.Random(140).shuffle(data)

trim_index = int(len(data) * 5/6)

train_data = data[:trim_index]
test_data = data[trim_index:]

# Lưu lại các tập 
from utility import dumpPickle

dumpPickle(train_data, 'before_train.pickle')
dumpPickle(test_data, 'test.pickle')