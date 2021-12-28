# Đọc tập train
import pickle
from time import time
from tfidfAtHome import tfidfAtHome

# pickleFile = open('.\\..\\data\\processed\\after_train.pickle', 'rb')
# train_data = pickle.load(pickleFile)
# pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\x_train.pkl', 'rb')
x_train_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\x_train.pkl', 'rb')
x_train_data_fake = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\y_train.pkl', 'rb')
y_train_data = pickle.load(pickleFile)
pickleFile.close()

tfidf = tfidfAtHome()
x_train_tfidf = tfidf.fitThenTransform(x_train_data_fake)

train_data = []
train_tfidf = []
for i in range(len(x_train_data)):
    train_data.append( (x_train_data[i], y_train_data[i]) )
    train_tfidf.append( (x_train_tfidf[i], y_train_data[i]) )

print(train_data[0])
print(train_tfidf[0])

# Train
from NaiveBayes import NaiBay
nbc = NaiBay()

start_time = time()
nbc.train(train_data)
# print('Train: ', time() - start_time)

nbc.dumpPickleSelf("after_train_TV.pickle")

nbc2 = NaiBay()
nbc2.train(train_tfidf)
nbc2.dumpPickleSelf("after_train_TV_tfidf.pickle")

