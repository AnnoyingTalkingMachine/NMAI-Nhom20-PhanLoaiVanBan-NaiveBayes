# Đọc tập train
import pickle
from time import time

# with open('.\\..\\data\\processed\\test.pickle', 'rb') as test:
#     test_data = pickle.load(test)

pickleFile = open('.\\..\\data\\VN\\x_test.pkl', 'rb')
x_train_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\y_test.pkl', 'rb')
y_train_data = pickle.load(pickleFile)
pickleFile.close()

test_data = []
for i in range(len(x_train_data)):
    test_data.append( (x_train_data[i], y_train_data[i]) )

from NaiveBayes import NaiBay
nbc = NaiBay()
nbc.loadPickleSelf("after_train.pickle")

start_time = time()
nbc.test(test_data)
print('Test: ', time() - start_time)
