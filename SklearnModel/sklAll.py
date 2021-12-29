###############
## Read data ##
###############
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
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
labels = ["POS", "NEU", "NEG"]
accuracy = [[], []]
confusionMatrix = [[], []]


#############################
## MULTINOMIAL NAIVE BAYES ##
#############################
print('MULTINOMIAL NAIVE BAYES')
from sklearn.naive_bayes import MultinomialNB
MulNB = MultinomialNB()
MulNB.fit(vectorTrainX, trainY)

pred_trainY = MulNB.predict(vectorTrainX)
pred_testY = MulNB.predict(vectorTestX)

accuracy[0].append(metrics.accuracy_score(trainY, pred_trainY))
accuracy[1].append(metrics.accuracy_score(testY, pred_testY))

confusionMatrix[0].append(metrics.confusion_matrix(trainY, pred_trainY, labels=labels))
confusionMatrix[1].append(metrics.confusion_matrix(testY, pred_testY, labels=labels))


#########################
## LOGISTIC REGRESSION ##
#########################
print('LOGISTIC REGRESSION')
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(vectorTrainX, trainY)

pred_trainY = logReg.predict(vectorTrainX)
pred_testY = logReg.predict(vectorTestX)

accuracy[0].append(metrics.accuracy_score(trainY, pred_trainY))
accuracy[1].append(metrics.accuracy_score(testY, pred_testY))

confusionMatrix[0].append(metrics.confusion_matrix(trainY, pred_trainY, labels=labels))
confusionMatrix[1].append(metrics.confusion_matrix(testY, pred_testY, labels=labels))


#########
## SVC ##
#########
print('SVC')
from sklearn.svm import SVC
svc = SVC()
svc.fit(vectorTrainX, trainY)

pred_trainY = svc.predict(vectorTrainX)
pred_testY = svc.predict(vectorTestX)

accuracy[0].append(metrics.accuracy_score(trainY, pred_trainY))
accuracy[1].append(metrics.accuracy_score(testY, pred_testY))

confusionMatrix[0].append(metrics.confusion_matrix(trainY, pred_trainY, labels=labels))
confusionMatrix[1].append(metrics.confusion_matrix(testY, pred_testY, labels=labels))


#######################################
## MULTI-LAYER PERCEPTRON CLASSIFIER ##
#######################################
print('MULTI-LAYER PERCEPTRON CLASSIFIER')
from sklearn.neural_network import MLPClassifier
mlpCsf = MLPClassifier()
mlpCsf.fit(vectorTrainX, trainY)

pred_trainY = mlpCsf.predict(vectorTrainX)
pred_testY = mlpCsf.predict(vectorTestX)

accuracy[0].append(metrics.accuracy_score(trainY, pred_trainY))
accuracy[1].append(metrics.accuracy_score(testY, pred_testY))

confusionMatrix[0].append(metrics.confusion_matrix(trainY, pred_trainY, labels=labels))
confusionMatrix[1].append(metrics.confusion_matrix(testY, pred_testY, labels=labels))


##################
## Show results ##
##################
for i in range(4):
    print("Train data:")
    print("Accuracy:", accuracy[0][i])
    print("Confusion Matrix:\n", confusionMatrix[0][i])
    print("Test data:")
    print("Accuracy:", accuracy[1][i])
    print("Confusion Matrix:\n", confusionMatrix[1][i])

fig, axes = plt.subplots(2, 4, figsize=(14,9))

sns.heatmap(confusionMatrix[0][0], annot=True, cmap='Blues', fmt='g', ax=axes[0, 0])
axes[0, 0].set_title("Multinomial Naive Bayes, dữ liệu Train\nAccuracy: " + str(accuracy[0][0]), fontsize=10)
sns.heatmap(confusionMatrix[1][0], annot=True, cmap='Blues', fmt='g', ax=axes[1, 0])
axes[1, 0].set_title("Multinomial Naive Bayes, dữ liệu Test\nAccuracy: " + str(accuracy[1][0]), fontsize=10)
sns.heatmap(confusionMatrix[0][1], annot=True, cmap='Reds', fmt='g', ax=axes[0, 1])
axes[0, 1].set_title("Logistic Regression, dữ liệu Train\nAccuracy: " + str(accuracy[0][1]), fontsize=10)
sns.heatmap(confusionMatrix[1][1], annot=True, cmap='Reds', fmt='g', ax=axes[1, 1])
axes[1, 1].set_title("Logistic Regression, dữ liệu Test\nAccuracy: " + str(accuracy[1][1]), fontsize=10)
sns.heatmap(confusionMatrix[0][2], annot=True, cmap='Oranges', fmt='g', ax=axes[0, 2])
axes[0, 2].set_title("Support Vector Classifier, dữ liệu Train\nAccuracy: " + str(accuracy[0][2]), fontsize=10)
sns.heatmap(confusionMatrix[1][2], annot=True, cmap='Oranges', fmt='g', ax=axes[1, 2])
axes[1, 2].set_title("Support Vector Classifier, dữ liệu Test\nAccuracy: " + str(accuracy[1][2]), fontsize=10)
sns.heatmap(confusionMatrix[0][3], annot=True, cmap='Greens', fmt='g', ax=axes[0, 3])
axes[0, 3].set_title("Multi Layer Perceptron Classifier, dữ liệu Train\nAccuracy: " + str(accuracy[0][3]), fontsize=10)
sns.heatmap(confusionMatrix[1][3], annot=True, cmap='Greens', fmt='g', ax=axes[1, 3])
axes[1, 3].set_title("Multi Layer Perceptron Classifier, dữ liệu Test\nAccuracy: " + str(accuracy[1][3]), fontsize=10)


plt.subplots_adjust(wspace=0.5, hspace=0.5)

for ax in axes.flat:
    ax.set(xlabel='Nhãn dự đoán', ylabel='Nhãn thực')
    ax.set(xticklabels=labels, yticklabels=labels)

plt.savefig("output\\All.png")
plt.show()