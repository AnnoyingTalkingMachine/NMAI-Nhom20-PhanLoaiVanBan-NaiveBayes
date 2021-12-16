# Đọc tập train
import pickle
from time import time

pickleFile = open('.\\..\\data\\processed\\before_train.pickle', 'rb')
train_data = pickle.load(pickleFile)
pickleFile.close()

# Train
from NaiveBayes import NaiBay
nbc = NaiBay()

start_time = time()
nbc.train(train_data)
print('Train: ', time() - start_time)

nbc.dumpPickleSelf("after_train.pickle")

