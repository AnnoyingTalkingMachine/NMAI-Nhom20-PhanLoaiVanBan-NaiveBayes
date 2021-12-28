import pickle

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('darkgrid') 

class NaiBay():
    def __init__(self):
        # Số dòng dữ liệu train
        self.lines = 0
        # List các từ phân biệt trong tất cả các câu
        self.distinctWords = {}
        # Tổng số lượng từ (không phân biệt)
        self.countWords = 0

        # Key: các nhãn phân biệt
        # Value: số lượng của các nhãn trong tập dữ liệu train
        self.distinctLabels = {}

        # Key: các nhãn phân biệt
        # Value: tổng số từ theo nhãn đó
        self.countAllWordByLabel = {}

        # Key: các từ phân biệt
        # Value: {subKey: các nhãn phân biệt, subValue: số lần xuất hiện từ này trong nhãn đó}
        self.countWordByLabel = {}

        # Key: các từ phân biệt
        # Value: {subKey: các nhãn phân biệt, subValue: tần xuất từ này trong nhãn đó}
        # -> P( word | label )
        self.probWordByLabel = {}
    # end __init__()


    ###################
    ##     TRAIN     ##
    ###################
    def train(self, train_data):
        print("Train")
        self.lines = len(train_data)

        for _, label in train_data:
            # Ghi lại key: label phân biệt, value: số lần xuất hiện 
            if label in self.distinctLabels:
                self.distinctLabels[label] += 1
            else:
                self.distinctLabels[label] = 1
                self.countAllWordByLabel[label] = 0

        for line, _ in train_data:
            # Ghi lại các từ phân biệt
            for word in line:
                self.countWords += line[word]
                self.distinctWords[word] = True
                self.countWordByLabel[word] = dict([label, 0] for label in self.distinctLabels)

        # Tính toán số lượng từ theo từng label
        for line, label in train_data:
            for word in line:
                self.countAllWordByLabel[label] += line[word]
                self.countWordByLabel[word][label] += line[word]

        # Tính P(word | label)
        for word in self.distinctWords:
            self.probWordByLabel[word] = {}

            for label in self.countWordByLabel[word]:
                numerator = self.countWordByLabel[word][label] + 1
                denominator = self.countAllWordByLabel[label] + len(self.distinctWords)
                
                self.probWordByLabel[word][label] = numerator / denominator
                
                if self.probWordByLabel[word][label] == 0: 
                    print("Từ này xác suất = 0:", word, label)
    # end train()


    ##################
    ##   CLASSIFY   ##
    ##################
    def classify(self, tokenized_sentence): 
        probLabel = {}
        for label in self.distinctLabels:
            probLabel[label] = 1

        # Nhân P(word | label) và xác suất
        for word in tokenized_sentence:
            # Nếu từ không xuất hiện trong dữ liệu train -> loại
            if word not in self.probWordByLabel: 
                continue
            
            for label in self.distinctLabels:
                temp = self.probWordByLabel[word][label] ** tokenized_sentence[word]
                probLabel[label] *= temp

        for label in self.distinctLabels:
            if probLabel[label] == 1 or probLabel[label] == 0:
                probLabel[label] = 0
                print("Câu này xác suất = 0", tokenized_sentence, label)

            temp = self.distinctLabels[label] #/ self.lines
            probLabel[label] *= temp

        # Chọn ra nhãn có xác suất cao nhất
        return max(probLabel, key=probLabel.get)
    # end classify()


                
    ##################
    ##     TEST     ##
    ##################
    def test(self, test_data, figureTitle=""):
        print("Test data")
        
        self.y_actu = []
        self.y_pred = []

        for tokenized_sentence, actu_label in test_data:
            pred_label = self.classify(tokenized_sentence)
            
            self.y_actu.append(actu_label)
            self.y_pred.append(pred_label)

        print(figureTitle)
        accuracy = metrics.accuracy_score(self.y_actu, self.y_pred)
        
        if len(self.distinctLabels) == 3:
            labels = ["POS", "NEU", "NEG"]
        elif len(self.distinctLabels) == 2:
            labels = [0, 1]
        else:
            labels = self.distinctLabels

        confusionMatrix =  metrics.confusion_matrix(self.y_actu, self.y_pred, labels=labels)

        print("Accuracy:", accuracy)
        print("Confusion matrix:\n", confusionMatrix)

        # ax = sns.heatmap(confusionMatrix, annot=True, cmap='Blues', fmt='g')
        # ax.set_title(figureTitle + "\nConfusion Matrix\n\n")
        # ax.set_xlabel('Nhãn dự đoán')
        # ax.set_ylabel('Nhãn thực')

        # ax.xaxis.set_ticklabels(labels)
        # ax.yaxis.set_ticklabels(labels)

        # ax.text(10, 20, "Accuracy:" + str(accuracy), ha ='center')
        # plt.show()

        return (confusionMatrix, figureTitle + "\nAccuracy = " + str(accuracy))
        
    # end test()


    def printThings(self):
        print("Print things")
        # print(self.lines)
        # print(self.distinctWords)
        # print(self.distinctLabels)
        # print(self.countAllWordByLabel)
        # print(self.countWordByLabel)

        # probWordByLabel
        # print(self.probWordByLabel)
        # for word in self.probWordByLabel:
        #     for label in self.probWordByLabel[word]:
        #         print("P(%s | %d) = %.6f" % (word, label, self.probWordByLabel[word][label]))
    # end printThing()

    def loadPickleSelf(self, fileName):
        pickleFile = open('.\\..\\data\\processed\\%s' % (fileName), 'rb')
        data = pickle.load(pickleFile)
        pickleFile.close()
        self.__dict__.update(data)

    def dumpPickleSelf(self, fileName):
        pickleFile = open('.\\..\\data\\processed\\%s' % (fileName), 'wb')
        pickle.dump(self.__dict__, pickleFile)
        pickleFile.close()

            






