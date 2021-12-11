from utility import *

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
        self.countWordInLine = []

        # Danh sách các nhãn của các dòng (theo ma trận trên)
        self.labels = []

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
            self.labels.append(label)

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
            self.countWordInLine.append([0] * self.countDistinctWords)
            for word in line:
                idx = self.distinctWords.index(word)
                self.countWordInLine[-1][idx] += line[word]
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
        
        # Lưu lại model đã train
        self.dumpPickleSelf()
    # end train()


    ##################
    ##   CLASSIFY   ##
    ##################
    def classify(self, tokenized_sentence):
        print("Classify")

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
                    print(word, label, temp)

        for label in self.distinctLabels:
            if probLabel[label] == 1:
                probLabel[label] = 0

            temp = self.distinctLabels[label] / len(self.labels)
            probLabel[label] *= temp
            print(label, temp)

        print('Label: ', probLabel)
        return max(probLabel, key=probLabel.get)
    # end classify()               
                

    ##################
    ##     TEST     ##
    ##################
    def test(self, test_data):
        print("Test data")

        confusionMatrix = {}
        
    # end test()


    def printThings(self):
        print("Print things")
        # print(self.distinctWords)
        # print(self.labels)
        # print(self.distinctLabels)
        # print(self.countWordInLine)
        # print(self.countAllWordByLabel)
        # print(self.countWordByLabel)

        # print(self.probWordByLabel)
        # # probWordByLabel
        # for word in self.probWordByLabel:
        #     for label in self.probWordByLabel[word]:
        #         print("P(%s | %d) = %.6f" % (word, label, self.probWordByLabel[word][label]))
    # end printThing()

    def loadPickleSelf(self):
        self.__dict__.update(loadPickle('after_train.pickle'))

    def dumpPickleSelf(self):
        dumpPickle(self.__dict__, 'after_train.pickle')

            






