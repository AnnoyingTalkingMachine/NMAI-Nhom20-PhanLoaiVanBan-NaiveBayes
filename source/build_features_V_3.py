# Chạy cái này chỉ để gộp mấy file của Nam lại

import pickle
import pandas as pd
from tfidfAtHome import tfidfAtHome

pickleFile = open('.\\..\\data\\VN\\x_test.pkl', 'rb')
x_test_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\y_test.pkl', 'rb')
y_test_data = pickle.load(pickleFile)
pickleFile.close()

test_data = []
x_test_data = x_test_data.to_list()
y_test_data = y_test_data.to_list()

for i in range(len(x_test_data)):
    test_data.append( (x_test_data[i], y_test_data[i]) )

pickleFile = open('.\\..\\data\\processed\\test_V_3.pickle', 'wb')
pickle.dump(test_data, pickleFile)
pickleFile.close()


pickleFile = open('.\\..\\data\\VN\\x_train.pkl', 'rb')
x_train_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\y_train.pkl', 'rb')
y_train_data = pickle.load(pickleFile)
pickleFile.close()

train_data = []
x_train_data = x_train_data.to_list()
y_train_data = y_train_data.to_list()

for i in range(len(x_train_data)):
    train_data.append( (x_train_data[i], y_train_data[i]) )

pickleFile = open('.\\..\\data\\processed\\before_train_V_3.pickle', 'wb')
pickle.dump(train_data, pickleFile)
pickleFile.close()
