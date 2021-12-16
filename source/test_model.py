# Đọc tập train
import pickle
from time import time

with open('.\\..\\data\\processed\\test.pickle', 'rb') as test:
    test_data = pickle.load(test)

from NaiveBayes import NaiBay
nbc = NaiBay()
nbc.loadPickleSelf()

start_time = time()

nbc.test(test_data)

print('Test: ', time() - start_time)