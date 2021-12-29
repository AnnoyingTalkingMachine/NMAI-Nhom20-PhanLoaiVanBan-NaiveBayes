# Đọc tập train
import pickle
from time import time

pickleFile = open('.\\..\\data\\processed\\before_train_E_3.pickle', 'rb')
train_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\processed\\before_train_E_3_tfidf.pickle', 'rb')
train_tfidf = pickle.load(pickleFile)
pickleFile.close()

# Train
from NaiveBayes_Logarit import NaiBay_L
nbc = NaiBay_L()

start_time = time()
nbc.train(train_data)
print('Train time: ', time() - start_time)

nbc.dumpPickleSelf("after_train_E_3.pickle")

nbc = NaiBay_L()
start_time = time()
nbc.train(train_tfidf)
print('Train time: ', time() - start_time)

nbc.dumpPickleSelf("after_train_tfidf_E_3.pickle")


