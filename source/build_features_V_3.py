# Data dataTV.csv của Nam

# Đọc dữ liệu raw
import pandas as pd
import random
import pickle

dataFrame = pd.read_csv("..\\data\\raw\\dataTV.csv", encoding="ISO-8859-1", header=None)
dataFrame.columns = ["comment", "rate"]
# dataFrame = dataFrame.sample(frac=1)

data_positive = []
data_negative = []
data_neutral = []

from tokenize_clean_data_V import text_preprocess
for idx, df in dataFrame.iterrows():
    try:
        temp = text_preprocess(df["comment"])
        if temp == {}:
            continue

        if df["rate"] > 3:
            data_positive.append( (temp, "POS") )
        elif df["rate"] < 3:
            data_negative.append( (temp, "NEG") )
        elif df["rate"] == 3:
            data_neutral.append( (temp, "NEU") )
    except:
        print(idx)

min_len = min(len(data_positive), len(data_neutral), len(data_negative))
data_positive = data_positive[:min_len]
data_neutral = data_neutral[:min_len]
data_negative = data_negative[:min_len]

data = data_positive + data_neutral + data_negative
print(len(data))

random.shuffle(data)
print(data[:10])

trim_index = int(len(data) * 0.8)

train_data = data[:trim_index]
test_data = data[trim_index:]

print(len(train_data), len(test_data))

pickleFile = open('.\\..\\data\\processed\\before_train_V_3.pickle', 'wb')
pickle.dump(train_data, pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\processed\\test_V_3.pickle', 'wb')
pickle.dump(test_data, pickleFile)
pickleFile.close()
