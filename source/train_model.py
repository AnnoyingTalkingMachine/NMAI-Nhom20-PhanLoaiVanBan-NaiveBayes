# Đọc tập train
from utility import *

train_data = loadPickle('before_train.pickle')

# Train
from NaiveBayes import NaiBay

nbc = NaiBay()
nbc.train(train_data)
nbc.printThings()


