# Chạy cái này chỉ để gộp mấy file của Nam lại

import pickle

from tfidfAtHome import tfidfAtHome

pickleFile = open('x_test.pkl', 'rb')
x_test_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('y_test.pkl', 'rb')
y_test_data = pickle.load(pickleFile)
pickleFile.close()

test_data = []
x_test_data = x_test_data.to_list()
y_test_data = y_test_data.to_list()

for i in range(len(x_test_data)):
    test_data.append( (x_test_data[i], y_test_data[i]) )

print(len(test_data))

pickleFile = open('.\\..\\processed\\test_V_3.pickle', 'wb')
pickle.dump(test_data, pickleFile)
pickleFile.close()


pickleFile = open('x_train.pkl', 'rb')
x_train_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('y_train.pkl', 'rb')
y_train_data = pickle.load(pickleFile)
pickleFile.close()

train_data = []
x_train_data = x_train_data.to_list()
y_train_data = y_train_data.to_list()

for i in range(len(x_train_data)):
    train_data.append( (x_train_data[i], y_train_data[i]) )

print(len(train_data))

pickleFile = open('.\\..\\processed\\before_train_V_3.pickle', 'wb')
pickle.dump(train_data, pickleFile)
pickleFile.close()


tfidf = tfidfAtHome()
train_tfidf = tfidf.fitThenTransform(train_data)
test_tfidf = tfidf.transform(test_data)

pickleFile = open('.\\..\\processed\\before_train_V_3_tfidf.pickle', 'wb')
pickle.dump(train_tfidf, pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\processed\\test_V_3_tfidf.pickle', 'wb')
pickle.dump(test_tfidf, pickleFile)
pickleFile.close()

