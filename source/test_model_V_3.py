# Đọc tập train
import pickle
from time import time
from tfidfAtHome import tfidfAtHome

# with open('.\\..\\data\\processed\\test.pickle', 'rb') as test:
#     test_data = pickle.load(test)

pickleFile = open('.\\..\\data\\VN\\x_test.pkl', 'rb')
x_test_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\x_test.pkl', 'rb')
x_test_data_fake = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\y_test.pkl', 'rb')
y_test_data = pickle.load(pickleFile)
pickleFile.close()

tfidf = tfidfAtHome()
x_test_tfidf = tfidf.transform(x_test_data_fake)

test_data = []
test_tfidf = []
for i in range(len(x_test_data)):
    test_data.append( (x_test_data[i], y_test_data[i]) )
    test_tfidf.append( (x_test_tfidf[i], y_test_data[i]) )

from NaiveBayes import NaiBay
nbc = NaiBay()
nbc.loadPickleSelf("after_train_TV.pickle")

start_time = time()
nbc.test(test_data)
print('Test time: ', time() - start_time)

nbc2 = NaiBay()
nbc2.loadPickleSelf("after_train_TV_tfidf.pickle")
nbc2.test(test_tfidf)
