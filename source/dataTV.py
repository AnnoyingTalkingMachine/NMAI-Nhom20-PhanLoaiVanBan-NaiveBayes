# Chạy cái này chỉ để gộp mấy file của Nam lại

import pickle
from time import time
from tfidfAtHome import tfidfAtHome

pickleFile = open('.\\..\\data\\VN\\x_test.pkl', 'rb')
x_test_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\y_test.pkl', 'rb')
y_test_data = pickle.load(pickleFile)
pickleFile.close()

test_data = []
for i in range(len(x_test_data)):
    test_data.append( (x_test_data[i], y_test_data[i]) )

pickleFile = open('.\\..\\data\\processed\\test_TV.pickle', 'wb')
pickle.dump(test_data, pickleFile)
pickleFile.close()


pickleFile = open('.\\..\\data\\VN\\x_train.pkl', 'rb')
x_train_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\y_train.pkl', 'rb')
y_train_data = pickle.load(pickleFile)
pickleFile.close()

train_data = []
for i in range(len(x_train_data)):
    train_data.append( (x_train_data[i], y_train_data[i]) )

pickleFile = open('.\\..\\data\\processed\\before_train_TV.pickle', 'wb')
pickle.dump(train_data, pickleFile)
pickleFile.close()

