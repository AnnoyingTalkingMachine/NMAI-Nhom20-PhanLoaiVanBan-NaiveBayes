import os
import pickle

# Đọc data của Nam
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/VN'))
dataDir = os.getcwd()

# X là tập dữ liệu, Y là tập nhãn ứng với tập dữ liệu
trainX = pickle.load(open(os.path.join(dataDir, 'x_train.pkl'), 'rb'))
trainY = pickle.load(open(os.path.join(dataDir, 'y_train.pkl'), 'rb'))

testX = pickle.load(open(os.path.join(dataDir, 'x_test.pkl'), 'rb'))
testY = pickle.load(open(os.path.join(dataDir, 'y_test.pkl'), 'rb'))


# Chuẩn hóa lại dữ liệu để đưa vào TFIDFVectorizer
# { "work": 3, "give": 1, "up": 1 } ---> "work work work give up"
dataTrainX = []
for data in trainX:
    temp = []
    for key in data:
        for i in range(data[key]):
            temp.append(str(key))
    temp = ' '.join(temp)
    dataTrainX.append(temp)

dataTestX = []
for data in testX:
    temp = []
    for key in data:
        for i in range(data[key]):
            temp.append(str(key))
    temp = ' '.join(temp)
    dataTestX.append(temp)

# TFIDF vào việc
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfProMax = TfidfVectorizer()
vectorTrainX = tfidfProMax.fit_transform(dataTrainX)
vectorTestX = tfidfProMax.transform(dataTestX)

print('Train Matrix Size:', vectorTrainX.shape)
print('Test Matrix Size:', vectorTestX.shape)

# Mã hóa nhãn để đưa vào MultinomialNB
from sklearn.preprocessing import LabelEncoder
lE = LabelEncoder()
encodedTrainY = lE.fit_transform(trainY)
encodedTestY = lE.transform(testY)

# MultinomialNB vào việc
from sklearn.naive_bayes import MultinomialNB
naibayProMax = MultinomialNB()
naibayProMax.fit(vectorTrainX, encodedTrainY)

print('Train Accuracy:', naibayProMax.score(vectorTrainX, encodedTrainY))
print('Test Accuracy:', naibayProMax.score(vectorTestX, encodedTestY))


# Hàm phân loại cảm xúc văn bản nhập vào từ bàn phím
import gensim
from pyvi import ViTokenizer

def phanLoaiVanBan():
    print("Nhập văn bản muốn phân loại:")  
    noiDung = input()
    tokens = ' '.join(noiDung)
    tokens = gensim.utils.simple_preprocess(noiDung)
    tokens = ' '.join(tokens)
    tokens = ViTokenizer.tokenize(tokens)
    tokens = [tokens]
    vtr = tfidfProMax.transform(tokens)
    encodedResult = naibayProMax.predict(vtr)
    ketqua = lE.inverse_transform(encodedResult)
    print(ketqua)

for i in range(5):
    phanLoaiVanBan()