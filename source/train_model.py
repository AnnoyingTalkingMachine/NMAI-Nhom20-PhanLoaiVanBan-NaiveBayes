# Đọc tập train
import pickle

with open('.\\..\\data\\processed\\before_train.pickle', 'rb') as for_train:
    train_data = pickle.load(for_train)


# Train
print("start NaiBay")
from NaiveBayes import NaiBay

classifier = NaiBay()
classifier.train(train_data)
classifier.printThings()