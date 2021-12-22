# Đọc tập train
import pickle
from time import time

# pickleFile = open('.\\..\\data\\processed\\after_train.pickle', 'rb')
# train_data = pickle.load(pickleFile)
# pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\x_train.pkl', 'rb')
x_train_data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open('.\\..\\data\\VN\\y_train.pkl', 'rb')
y_train_data = pickle.load(pickleFile)
pickleFile.close()

train_data = []
for i in range(len(x_train_data)):
    train_data.append( (x_train_data[i], y_train_data[i]) )

# Train
from NaiveBayes import NaiBay
nbc = NaiBay()

start_time = time()
nbc.train(train_data)
print('Train: ', time() - start_time)

nbc.dumpPickleSelf("after_train.pickle")

nbc.printThings()

