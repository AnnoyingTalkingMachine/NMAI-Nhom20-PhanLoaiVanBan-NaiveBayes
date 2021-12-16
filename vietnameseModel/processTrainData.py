from vneseDataProcess import text_preprocess

import pandas as pd
import numpy as np

df_raw = pd.read_csv('./input/dataVNese.csv', encoding = "UTF-8")

# lấy số lượng các câu ở mỗi label bằng nhau (4700 câu/1 label) #
df_raw.drop(df_raw.iloc[:, 3:], axis = 1, inplace = True)

df_raw_neg = df_raw[df_raw['label'] == 'NEG' ]
df_raw_neg = df_raw_neg.iloc[:4700]

df_raw_neu = df_raw[df_raw['label'] == 'NEU']

df_raw_4 = df_raw[df_raw['rate'] == 4]
df_raw_5 = df_raw[df_raw['rate'] == 5]

df_raw_4 = df_raw_4.iloc[:2350]
df_raw_5 = df_raw_5.iloc[:2350]

df_raw_pos = pd.concat([df_raw_4, df_raw_5])
dataraw_final = pd.concat([df_raw_neg, df_raw_neu, df_raw_pos])

dataraw_final = dataraw_final.sort_index()

# xử lý dữ liệu train #
process_comment = []
comment = dataraw_final.comment
for item in comment:
    a = text_preprocess(item)
    process_comment.append(a)

list_tuples = list(zip(process_comment, dataraw_final.label)) 
processed_data = pd.DataFrame(list_tuples, columns = ['comment', 'label'])

x_train = processed_data['comment']
y_train = processed_data['label']

#lưu file x_train và y_train#
import pickle

with open('x_train.pkl', 'wb') as f:
     pickle.dump(x_train, f)
with open('y_train.pkl', 'wb') as f:
     pickle.dump(y_train, f)  





