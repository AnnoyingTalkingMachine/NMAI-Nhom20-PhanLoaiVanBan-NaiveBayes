import pickle
from plot_confusion_matrix import plot_confusion_matrix, to_df_confusion

class NaiBay():
    def __init__(self):
        # List các từ phân biệt trong tất cả các câu
        self.distinctWords = []
        # Tổng số lượng từ (phân biệt)
        self.countDistinctWords = 0
        # Tổng số lượng từ (không phân biệt)
        self.countWords = 0 

        # Ma trận gồm nhiều dòng, mỗi dòng là 1 câu trong tập dữ liệu train
        # Mỗi cột trên một dòng sẽ là số lần xuất hiện của từ cụ thể trong câu
        # Thứ tự các từ đại diện các cột trùng thứ tự tring distinctWords 
        # self.countWordInLine = []

        # Danh sách các nhãn của các dòng (theo ma trận trên)
        # self.labels = []
        # Tổng số nhãn (= tổng số dòng dữ liệu)
        self.countLabels = 0

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

        for line, label in train_data:
            # Ghi lại toàn bộ label theo thứ tự xuất hiện trong tập train
            # self.labels.append(label)
            self.countLabels += 1

            # Ghi lại key: label phân biệt, value: số lần xuất hiện 
            if label in self.distinctLabels:
                self.distinctLabels[label] += 1
            else:
                self.distinctLabels[label] = 1
                self.countAllWordByLabel[label] = 0

            # Ghi lại các từ phân biệt
            for word in line:
                self.countWords += line[word]
                if word not in self.distinctWords:
                    self.distinctWords.append(word)
                    self.countDistinctWords += 1

        # Lập ma trận
        for line, label in train_data:
            # self.countWordInLine.append([0] * self.countDistinctWords)
            for word in line:
                # idx = self.distinctWords.index(word)
                # self.countWordInLine[-1][idx] += line[word]
                self.countAllWordByLabel[label] += line[word]
                
                if word in self.countWordByLabel:
                    if label in self.countWordByLabel[word]:
                        self.countWordByLabel[word][label] += line[word]
                    else:
                        self.countWordByLabel[word][label] = line[word]
                else:
                    self.countWordByLabel[word] = {}
                    self.countWordByLabel[word][label] = line[word]

        # Tính self.probWordByLabel
        for word in self.distinctWords:
            self.probWordByLabel[word] = {}

            for label in self.countWordByLabel[word]:
                numerator = self.countWordByLabel[word][label] + 1
                denominator = self.countAllWordByLabel[label] + self.countDistinctWords

                self.probWordByLabel[word][label] = numerator / denominator
    # end train()


    ##################
    ##   CLASSIFY   ##
    ##################
    def classify(self, tokenized_sentence):
        # print("Classify")

        probLabel = {}
        for label in self.distinctLabels:
            probLabel[label] = 1

        for word in tokenized_sentence:
            if word not in self.probWordByLabel: 
                continue
            
            for label in self.distinctLabels:
                if label in self.probWordByLabel[word]:
                    temp = self.probWordByLabel[word][label] ** tokenized_sentence[word]
                    probLabel[label] *= temp
                    # print(word, label, temp)

        for label in self.distinctLabels:
            if probLabel[label] == 1:
                probLabel[label] = 0

            temp = self.distinctLabels[label] / self.countLabels # len(self.labels)
            probLabel[label] *= temp

        # print('Label: ', probLabel)
        return max(probLabel, key=probLabel.get)
    # end classify()


                

    ##################
    ##     TEST     ##
    ##################
    def test(self, test_data):
        print("Test data")
        
        self.y_actu = []
        self.y_pred = []

        for tokenized_sentence, actu_label in test_data:
            pred_label = self.classify(tokenized_sentence)
            self.y_actu.append(actu_label)
            self.y_pred.append(pred_label)

        plot_confusion_matrix(to_df_confusion(self.y_actu, self.y_pred))

        # confusionMatrix = {}
        # for label1 in self.distinctLabels:
        #     confusionMatrix[label1] = {}
        #     for label2 in self.distinctLabels:
        #         confusionMatrix[label1][label2] = 0
        
        # for tokenized_sentence, actu_label in test_data:
        #     pred_label = self.classify(tokenized_sentence)
        #     confusionMatrix[actu_label][pred_label] += 1

        # correct_pred = 0
        # incorrect_pred = 0
        # for label1 in confusionMatrix:
        #     for label2 in confusionMatrix[label1]:
        #         if label1 == label2:
        #             correct_pred += confusionMatrix[label1][label2]
        #         else:
        #             incorrect_pred += confusionMatrix[label1][label2]
        # print("Accuracy: %f" % ( correct_pred / (correct_pred + incorrect_pred) ) )
        
        # print(confusionMatrix)

        #SKLEARN    
    # end test()


    def printThings(self):
        print("Print things")
        print(self.distinctWords)
        print(self.distinctLabels)
        # print(self.countWordInLine)
        print(self.countAllWordByLabel)
        print(self.countWordByLabel)

        # print(self.probWordByLabel)
        # probWordByLabel
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

            






