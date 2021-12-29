# Đọc dữ liệu raw
import pandas as pd
from time import time
import random
from tfidfAtHome import tfidfAtHome

dataFrame_raw = pd.read_csv("..\\data\\raw\\dataset.csv", encoding="ISO-8859-1", header=None)
dataFrame_raw.columns = ["label", "time", "date", "query", "username", "text"]

dataFrame = dataFrame_raw[["label", "text"]]
# dataFrame = dataFrame.sample(frac=1)
# Cắt nhỏ kích thước dữ liệu
start_time = time()

dataFrame_positive = dataFrame[dataFrame["label"] == 4]
dataFrame_negative = dataFrame[dataFrame["label"] == 0]

dataFrame_positive = dataFrame_positive.iloc[:100000] # :int(len(dataFrame_positive) / 40)
dataFrame_negative = dataFrame_negative.iloc[:100000] # :int(len(dataFrame_negative) / 40)

dataFrame = pd.concat([dataFrame_positive, dataFrame_negative])

print('Cắt dữ liệu: ', time() - start_time)

# [ (text, label) ] -> [ ( {token: số lần lặp}, label ) ]
from tokenize_clean_data_E import tokenize_clean_sentence

start_time = time()

data = []
for index, df in dataFrame.iterrows():
    try:
        temp = tokenize_clean_sentence(df["text"])
        if not temp:
            print(index, df["text"], "=> removed")
            continue

        if df["label"] == 4:
            data.append( (temp, 1) )
        else:
            data.append( (temp, 0) )
    
    except:
        print(index, df["text"], "=> can't be cleaned")

print('Làm sạch dữ liệu: ', time() - start_time)

# Chia thành tập train và test


start_time = time()

random.shuffle(data)

trim_index = int(len(data) * 0.8)

train_data = data[:trim_index]
test_data = data[trim_index:]

print(len(train_data), len(test_data))
print('Chia tập: ', time() - start_time)

# Lưu lại các tập 
import pickle

start_time = time()

pickleFile = open('.\\..\\data\\processed\\before_train_E_2.pickle', 'wb')
pickle.dump(train_data, pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\processed\\test_E_2.pickle', 'wb')
pickle.dump(test_data, pickleFile)
pickleFile.close()

tfidf = tfidfAtHome()
train_tfidf = tfidf.fitThenTransform(train_data)
test_tfidf = tfidf.transform(test_data)

pickleFile = open('.\\..\\data\\processed\\before_train_E_2_tfidf.pickle', 'wb')
pickle.dump(train_tfidf, pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\processed\\test_E_2_tfidf.pickle', 'wb')
pickle.dump(test_tfidf, pickleFile)
pickleFile.close()

print('Lưu dữ liệu: :', time() - start_time)