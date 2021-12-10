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

    # Đưa dữ liệu train vào các biến thuộc tính (bên trên)
    def pre_train(self, data):
        for line, label in data:
            self.labels.append(label)

            if label in self.distinctLabels:
                self.distinctLabels[label] += 1
            else:
                self.distinctLabels[label] = 1
                self.countAllWordByLabel[label] = 0

            for word in line:
                self.countWords += line[word]
                if word not in self.distinctWords:
                    self.distinctWords.append(word)
                    self.countDistinctWords += 1

        for line, label in data:
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
    # end pre_train()

    # Tính self.prob
    def train(self, train_data):
        self.pre_train(train_data)

        for word in self.distinctWords:
            self.probWordByLabel[word] = {}
            for label in self.countWordByLabel[word]:
                numerator = self.countWordByLabel[word][label] + 1
                denominator = self.countAllWordByLabel[label] + self.countDistinctWords
                self.probWordByLabel[word][label] = numerator / denominator
    # end train()

    def classify(self, sentence):
        print("Classify")
    # end classify()               
                

    def test(test_data):
        print("Test data")
    # end test()


    def printThings(self):
        # print(self.distinctWords)
        # print(self.labels)
        # print(self.distinctLabels)
        # print(self.countWordInLine)
        # print(self.countAllWordByLabel)
        # print(self.countWordByLabel)
        for word in self.probWordByLabel:
            print(word)
            for label in self.probWordByLabel[word]:
                print(label, ": ", self.probWordByLabel[word][label])

    

            






