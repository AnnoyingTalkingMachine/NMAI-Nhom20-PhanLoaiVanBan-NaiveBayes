# Đọc tập train
from utility import *
from time import time

train_data = loadPickle('before_train.pickle')

# Train
from NaiveBayes import NaiBay
nbc = NaiBay()

start_time = time()

nbc.train(train_data)

print('Train: ', time() - start_time)

