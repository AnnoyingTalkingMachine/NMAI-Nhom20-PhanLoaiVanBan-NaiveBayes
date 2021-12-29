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

from sklearn.preprocessing import LabelEncoder
labelE = LabelEncoder()
trainY = labelE.fit_transform(trainY)
testY = labelE.transform(testY)

from sklearn import metrics
testResults = []

#############################
## MULTINOMIAL NAIVE BAYES ##
#############################
from sklearn.naive_bayes import MultinomialNB
MulNB = MultinomialNB()
MulNB.fit(vectorTrainX, trainY)

print('MULTINOMIAL NAIVE BAYES')
print('Train Accuracy:', MulNB.score(vectorTrainX, trainY))
print('Test Accuracy:', MulNB.score(vectorTestX, testY))


#########################
## LOGISTIC REGRESSION ##
#########################
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(vectorTrainX, trainY)

print('LOGISTIC REGRESSION')
print('Train Accuracy:', logReg.score(vectorTrainX, trainY))
print('Test Accuracy:', logReg.score(vectorTestX, testY))


#########
## SVC ##
#########
from sklearn.svm import SVC
mlpCsf = SVC()
mlpCsf.fit(vectorTrainX, trainY)

print('SVC')
print('Train Accuracy:', mlpCsf.score(vectorTrainX, trainY))
print('Test Accuracy:', mlpCsf.score(vectorTestX, testY))


#######################################
## MULTI-LAYER PERCEPTRON CLASSIFIER ##
#######################################
from sklearn.neural_network import MLPClassifier
mlpCsf = MLPClassifier()
mlpCsf.fit(vectorTrainX, trainY)

print('MULTI-LAYER PERCEPTRON CLASSIFIER')
print('Train Accuracy:', mlpCsf.score(vectorTrainX, trainY))
print('Test Accuracy:', mlpCsf.score(vectorTestX, testY))


