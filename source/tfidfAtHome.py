import math


# Remember to write comments properly!
# Luminescence

# Mô hình TFIDF này làm việc trên data là list các dictionary có dạng {token : số_lần_xuất_hiện}


class tfidfAtHome():
    # reset() mới là hàm init thật sự (Cảm ơn StackOverflow)
    def __init__(self):
        self.reset()
    def reset(self):
        # Từ điển
        self.corpus = set()


    # Tạo một từ điển mới dựa theo tập trainData
    """
    def fit(self, trainData):
        #Xóa từ điển cũ
        self.reset()

        for item in trainData:
            tempList = []
            for key in item:
                tempList.append(key)
            self.corpus.update(tempList)
    """
    def fit(self, trainData):
        #Xóa từ điển cũ
        self.reset()

        for tokens, label in trainData:
            tempList = []
            for key in tokens:
                tempList.append(key)
            self.corpus.update(tempList)


    # Vector hóa targetData theo từ điển đã fit
    """
    def transform(self, targetData):
        vectors = [] # kết quả trả về
        
        # Tính IDF trên từng từ trong từ điển
        idf = dict.fromkeys(self.corpus, 1)
        docCount = len(targetData)
        for item in targetData:
            for key in item:
                if key in self.corpus:
                    idf[key] += 1
        
        for key in idf:
            idf[key] = math.log10(docCount/idf[key])
            # idf[key] = (idf[key]/docCount)

        #Tính TFIDF cho từng văn bản
        for item in targetData:
            wordCount = 0

            # vector = dict.fromkeys(self.corpus, 0)
            # for key, value in item.items():
            #     wordCount = wordCount + value
            #     if key in self.corpus:
            #         vector[key] = value

            vector = item
            for key in vector:
                wordCount += vector[key]
            
            # Tính TFIDF
            for key in vector:
                if key in self.corpus:
                    vector[key] = vector[key]/wordCount*idf[key]
            
            vectors.append(vector)
        
        return vectors
    """

    def transform(self, targetData):
        vectors = [] # kết quả trả về
        
        # Tính IDF trên từng từ trong từ điển
        idf = dict.fromkeys(self.corpus, 1)
        docCount = len(targetData)
        for tokens, label in targetData:   
            for key in tokens:
                if key in self.corpus:
                    idf[key] += 1
        
        for key in idf:
            idf[key] = math.log10(docCount/idf[key])
            # idf[key] = (idf[key]/docCount)

        #Tính TFIDF cho từng văn bản
        for tokens, label in targetData:
            wordCount = 0

            # vector = dict.fromkeys(self.corpus, 0)
            # for key, value in item.items():
            #     wordCount = wordCount + value
            #     if key in self.corpus:
            #         vector[key] = value

            vector = tokens.copy()
            for key in vector:
                wordCount += vector[key]
            
            # Tính TFIDF
            for key in vector:
                if key in self.corpus:
                    vector[key] = vector[key]/wordCount*idf[key]
            
            vectors.append((vector, label))
        
        return vectors

    def fitThenTransform(self, targetData):
        self.fit(targetData)
        vectors = self.transform(targetData)
        return vectors



