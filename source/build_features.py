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


# Token hóa
from nltk.tokenize import TweetTokenizer

tk = TweetTokenizer(reduce_len=True)
data = []

for index, df in dataFrame.iterrows():
    if df["label"] == 4:
        data.append( (tk.tokenize(df["text"]), 1) )
    else:
        data.append( (tk.tokenize(df["text"]), 0) )


# Làm sạch token
from clean_data import clean_tokens

cleaned_token_list = []
for tokens, label in data:
    cleaned_token_list.append( (clean_tokens(tokens), label) )


# Rút gọn các token trùng nhau
def list_to_dict(cleaned_tokens):
    myDict = dict()

    for token in cleaned_tokens:
        if token in myDict:
            myDict[token] += 1
        else:
            myDict[token] = 1

    return myDict

final_data = []
for tokens, label in cleaned_token_list:
    final_data.append( (list_to_dict(tokens), label))


# Chia thành tập train và test
import random

random.Random(140).shuffle(final_data)

trim_index = int(len(final_data) * 1)

train_data = final_data[:trim_index]
test_data = final_data[trim_index:]


# Lưu lại các tập 
import pickle

with open('.\\..\\data\\processed\\before_train.pickle', 'wb') as for_train:
    pickle.dump(train_data, for_train)

with open('.\\..\\data\\processed\\test.pickle', 'wb') as for_test:
    pickle.dump(train_data, for_test)   