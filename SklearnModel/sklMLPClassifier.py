import pickle

with open('.\\..\\data\\processed\\before_train_V_3.pickle', 'rb') as train:
    train_data = pickle.load(train)

with open('.\\..\\data\\processed\\test_V_3.pickle', 'rb') as test:
    test_data = pickle.load(test)

dataTrainX = []
trainY = []
for tokens, label in train_data:
    dataTrainX.append(tokens)
    trainY.append(label)

dataTestX = []
testY = []
for tokens, label in test_data:
    dataTestX.append(tokens)
    testY.append(label)

from sklearn.feature_extraction import DictVectorizer
dictV = DictVectorizer()
vectorTrainX = dictV.fit_transform(dataTrainX)
vectorTestX = dictV.transform(dataTestX)

from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
labels = ["POS", "NEU", "NEG"]

from sklearn.neural_network import MLPClassifier
mlpCsf = MLPClassifier()
mlpCsf.fit(vectorTrainX, trainY)

pred_trainY = mlpCsf.predict(vectorTrainX)
pred_testY = mlpCsf.predict(vectorTestX)

train_accuracy = metrics.accuracy_score(trainY, pred_trainY)
test_accuracy = metrics.accuracy_score(testY, pred_testY)

train_confusionMatrix = metrics.confusion_matrix(trainY, pred_trainY, labels=labels)
test_confusionMatrix = metrics.confusion_matrix(testY, pred_testY, labels=labels)

fig, axes = plt.subplots(1, 2, figsize=(14,9), sharex=True, sharey=True)

sns.heatmap(train_confusionMatrix, annot=True, cmap='Greens', fmt='g', ax=axes[0])
axes[0].set_title("Multi Layer Perceptron Classifier, dữ liệu Train\nAccuracy: " + str(train_accuracy))
sns.heatmap(test_confusionMatrix, annot=True, cmap='Greens', fmt='g', ax=axes[1])
axes[1].set_title("Multi Layer Perceptron Classifier, dữ liệu Test\nAccuracy: " + str(test_accuracy))

plt.subplots_adjust(wspace=0.5, hspace=0.5)
for ax in axes.flat:
    ax.set(xlabel='Nhãn dự đoán', ylabel='Nhãn thực')
    ax.set(xticklabels=labels, yticklabels=labels)

plt.savefig("output\\MLPClassifier.png")
plt.show()