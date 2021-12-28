# Data text_emotion.csv của Đức Anh

# Đọc dữ liệu raw
import pandas as pd
import random
import pickle

dataFrame_raw = pd.read_csv("..\\data\\raw\\text_emotion.csv", encoding="ISO-8859-1", header=None)
dataFrame_raw.columns = ["tweet_id", "sentiment", "author", "content"]

data = {}
for idx, df in dataFrame_raw.iterrows():
    data[df["sentiment"]] = 1

print(data)
dataFrame = dataFrame_raw[["sentiment", "content"]]

data_positive = []
data_negative = []
data_neutral = []

from tokenize_clean_data import tokenize_clean_sentence
for idx, df in dataFrame.iterrows():
    try:
        temp = tokenize_clean_sentence(df["content"])
        if temp == {}:
            continue

        if df["sentiment"] in ["enthusiasm", "surprise", "love", "fun", "happiness", "relief"]:
            data_positive.append( (temp, "POS") )
        elif df["sentiment"] in ["empty", "sadness", "worry", "hate", "boredom", "anger"]:
            data_negative.append( (temp, "NEG") )
        elif df["sentiment"] == "neutral":
            data_neutral.append( (temp, "NEU") )
    except:
        print(idx)

print(len(data_positive), len(data_neutral), len(data_negative))
data = data_positive + data_neutral + data_negative
print(len(data))

random.shuffle(data)
print(data[:10])

trim_index = int(len(data) * 0.8)

train_data = data[:trim_index]
test_data = data[trim_index:]

print(len(train_data), len(test_data))

pickleFile = open('.\\..\\data\\processed\\before_train_new_E.pickle', 'wb')
pickle.dump(train_data, pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\processed\\test_new_E.pickle', 'wb')
pickle.dump(test_data, pickleFile)
pickleFile.close()
