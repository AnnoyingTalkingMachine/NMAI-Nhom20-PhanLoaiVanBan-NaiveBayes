from vneseDataProcess import text_preprocess

import pandas as pd
import numpy as np

data_test_raw = pd.read_csv('./input/test_concat.csv', encoding='utf-8')
data_test_raw.drop(data_test_raw.columns[2], axis= 1, inplace=True)

# Gán nhãn dữ liệu #
label = []
for item in data_test_raw['True_Label']:
    if(item == 5 or item == 4): 
        a = "POS"
        label.append(a)
    elif (item == 3):
        a = "NEU"
        label.append(a)
    else:
        a = "NEG"
        label.append(a)


data_test_raw['label'] = label

data_test_raw.drop(data_test_raw.columns[1], axis = 1, inplace = True)

# lấy số lượng câu ở mỗi nhãn là như nhau#
test_raw_neg = data_test_raw[data_test_raw['label'] == 'NEG' ]

test_raw_neu = data_test_raw[data_test_raw['label'] == 'NEU']

test_raw_pos = data_test_raw[data_test_raw['label'] == 'POS']

test_raw_pos = test_raw_pos.iloc[:705]
test_raw_neg = test_raw_neg.iloc[:705]

test_raw_final = pd.concat([test_raw_pos, test_raw_neu, test_raw_neg])

test_raw_final = test_raw_final.sort_index()

# xử lý dữ liệu test#
test_comment = []
comment = test_raw_final.comment
for item in comment:
    a = text_preprocess(item)
    test_comment.append(a)

list_tuples2 = list(zip(test_comment, test_raw_final.label)) 
processed_data = pd.DataFrame(list_tuples2, columns = ['comment', 'label'])


x_test = processed_data['comment']
y_test = processed_data['label']

# lưu file x_test và y_test  #
import pickle
with open('x_test.pkl', 'wb') as f:
     pickle.dump(x_test, f)
with open('y_test.pkl', 'wb') as f:
     pickle.dump(y_test, f) 