# Đọc tập train
import pickle
from tfidfAtHome import tfidfAtHome

with open('.\\..\\data\\processed\\test.pickle', 'rb') as test:
    test_data = pickle.load(test)

with open('.\\..\\data\\processed\\test.pickle', 'rb') as test:
    test_tfidf = pickle.load(test)

with open('.\\..\\data\\processed\\before_train.pickle', 'rb') as train:
    pretrain_data = pickle.load(train)
    
with open('.\\..\\data\\processed\\before_train.pickle', 'rb') as train:
    pretrain_tfidf = pickle.load(train)

tfidfAtHome()
tfidf = tfidfAtHome()
# test_tfidf[:] = tfidf.transform([test_data[:][0]])
# pretrain_tfidf[:] = tfidf.transform(pretrain_data[:][0])

from NaiveBayes import NaiBay
nbc = NaiBay()
nbc.loadPickleSelf("after_train.pickle")

print("Sử dụng đếm:")
nbc.test(pretrain_data, "Dữ liệu train sử dụng đếm")
nbc.test(test_data, "Dữ liệu test sử dụng đếm")

# print("Sử dụng TF-IDF: (Cái này chưa áp dụng)")
# nbc.test(pretrain_tfidf, "Dữ liệu train sử dụng TF-IDF")
# nbc.test(test_tfidf, "Dữ liệu test sử dụng TF-IDF")