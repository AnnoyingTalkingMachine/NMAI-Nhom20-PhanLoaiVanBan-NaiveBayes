# Đọc tập train
import pickle
from tfidfAtHome import tfidfAtHome
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

test_fileName = 'test.pickle'
beforeTrain_fileName = 'before_train.pickle'
afterTrain_fileName = 'after_train.pickle'
afterTrain_tfidf_fileName = '"after_train_tfidf.pickle"'

# Load dữ liệu
with open('.\\..\\data\\processed\\' + test_fileName, 'rb') as test:
    test_data = pickle.load(test)

with open('.\\..\\data\\processed\\' + test_fileName, 'rb') as test:
    test_tfidf = pickle.load(test)

with open('.\\..\\data\\processed\\' + beforeTrain_fileName, 'rb') as train:
    pretrain_data = pickle.load(train)
    
with open('.\\..\\data\\processed\\' + beforeTrain_fileName, 'rb') as train:
    pretrain_tfidf = pickle.load(train)

tfidfAtHome()
tfidf = tfidfAtHome()
test_tfidf = tfidf.transform(test_data)
pretrain_tfidf = tfidf.transform(pretrain_data)


testResults = []
# Test các data sử dụng đếm
from NaiveBayes_Logarit import NaiBay_L
nbc = NaiBay_L()
nbc.loadPickleSelf(afterTrain_fileName)

print("Sử dụng đếm:")
testResults.append(nbc.test(pretrain_data, "Dữ liệu train sử dụng đếm"))
testResults.append(nbc.test(test_data, "Dữ liệu test sử dụng đếm"))

# Test các data sử dụng TF-IDF
nbc = NaiBay_L()
nbc.loadPickleSelf(afterTrain_tfidf_fileName)

print("Sử dụng TF-IDF:")
testResults.append(nbc.test(pretrain_tfidf, "Dữ liệu train sử dụng TF-IDF"))
testResults.append(nbc.test(test_tfidf, "Dữ liệu test sử dụng TF-IDF"))

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(15,15), sharex=True, sharey=True)

sns.heatmap(testResults[0][0], annot=True, cmap='Blues', fmt='g', ax=axes[0, 0])
axes[0, 0].set_title(testResults[0][1])
sns.heatmap(testResults[1][0], annot=True, cmap='Reds', fmt='g', ax=axes[0, 1])
axes[0, 1].set_title(testResults[1][1])
sns.heatmap(testResults[2][0], annot=True, cmap='Oranges', fmt='g', ax=axes[1, 0])
axes[1, 0].set_title(testResults[2][1])
sns.heatmap(testResults[3][0], annot=True, cmap='Greens', fmt='g', ax=axes[1, 1])
axes[1, 1].set_title(testResults[3][1])

plt.subplots_adjust(wspace=0.5, hspace=0.5)

for ax in axes.flat:
    ax.set(xlabel='Nhãn dự đoán', ylabel='Nhãn thực')

for ax in axes.flat:
     ax.label_outer()

plt.show()