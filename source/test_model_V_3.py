# Đọc tập train
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load dữ liệu
with open('.\\..\\data\\processed\\test_V_3.pickle', 'rb') as test:
    test_data = pickle.load(test)

with open('.\\..\\data\\processed\\test_V_3_tfidf.pickle', 'rb') as test:
    test_tfidf = pickle.load(test)

with open('.\\..\\data\\processed\\before_train_V_3.pickle', 'rb') as train:
    pretrain_data = pickle.load(train)
    
with open('.\\..\\data\\processed\\before_train_V_3_tfidf.pickle', 'rb') as train:
    pretrain_tfidf = pickle.load(train)


testResults = []
# Test các data sử dụng đếm
from NaiveBayes_Logarit import NaiBay_L
nbc = NaiBay_L()
nbc.loadPickleSelf('after_train_V_3.pickle')

print("Sử dụng đếm:")
testResults.append(nbc.test(pretrain_data, "Dữ liệu train sử dụng đếm"))
testResults.append(nbc.test(test_data, "Dữ liệu test sử dụng đếm"))

# Test các data sử dụng TF-IDF
nbc = NaiBay_L()
nbc.loadPickleSelf('after_train_tfidf_V_3.pickle')

print("Sử dụng TF-IDF:")
testResults.append(nbc.test(pretrain_tfidf, "Dữ liệu train sử dụng TF-IDF"))
testResults.append(nbc.test(test_tfidf, "Dữ liệu test sử dụng TF-IDF"))

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(10,6))

sns.heatmap(testResults[0][0], annot=True, cmap='Blues', fmt='g', ax=axes[0, 0])
axes[0, 0].set_title(testResults[0][1])
sns.heatmap(testResults[1][0], annot=True, cmap='Reds', fmt='g', ax=axes[1, 0])
axes[1, 0].set_title(testResults[1][1])
sns.heatmap(testResults[2][0], annot=True, cmap='Oranges', fmt='g', ax=axes[0, 1])
axes[0, 1].set_title(testResults[2][1])
sns.heatmap(testResults[3][0], annot=True, cmap='Greens', fmt='g', ax=axes[1, 1])
axes[1, 1].set_title(testResults[3][1])

plt.subplots_adjust(wspace=0.5, hspace=0.5)

for ax in axes.flat:
    ax.set(xlabel='Nhãn dự đoán', ylabel='Nhãn thực')
    ax.set(xticklabels=testResults[0][2], yticklabels=testResults[0][2])

plt.savefig("output\\V_3.png")
plt.show()