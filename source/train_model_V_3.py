# Đọc tập train
import pickle
from time import time

pickleFile = open('.\\..\\data\\processed\\before_train_V_3.pickle', 'rb')
train_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\processed\\before_train_V_3.pickle', 'rb')
train_tfidf = pickle.load(pickleFile)
pickleFile.close()

from tfidfAtHome import tfidfAtHome
tfidf = tfidfAtHome()
train_tfidf = tfidf.fitThenTransform(train_tfidf)

# Train
from NaiveBayes_Logarit import NaiBay_L
nbc = NaiBay_L()

start_time = time()
nbc.train(train_data)
print('Train time: ', time() - start_time)

nbc.dumpPickleSelf("after_train_V_3.pickle")

nbc = NaiBay_L()
start_time = time()
nbc.train(train_tfidf)
print('Train time: ', time() - start_time)

nbc.dumpPickleSelf("after_train_tfidf_V_3.pickle")


