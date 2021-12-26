import pickle
from tkinter import *
from tkinter.scrolledtext import ScrolledText

from NaiveBayes import NaiBay
# import làm sạch TV ở đây
# ...
from tokenize_clean_data import tokenize_clean_sentence

nbc_V = NaiBay()
nbc_V.loadPickleSelf("after_train_TV.pickle")
nbc_E = NaiBay()
nbc_E.loadPickleSelf("after_train.pickle")

nbc_default = nbc_E
preTrain_filePath_default = '.\\..\\data\\processed\\before_train.pickle'
test_filePath_default = '.\\..\\data\\processed\\test.pickle'
aftTrain_fileName_default = "after_train.pickle"
tokenize_clean_default = tokenize_clean_sentence

############
## WINDOW ##
############
mainWindow = Tk()
mainWindow.title("Sentiment Classifier - Nhom 20 - Nhap mon AI")
mainWindow.geometry("500x400")

###########
## LABEL ##
###########
label = Label(mainWindow, text="Type something")
label.pack()
label.config(font=("Courier", 30))

###########
## INPUT ##
###########
sentenceInput = ScrolledText(mainWindow)
sentenceInput.place(x=30, y=50, width=440, height=100)
sentenceInput.config(font=("Courier", 15))

result = Label(mainWindow, text="", font=("Courier", 15))
result.place(x=150, y=200, width=200, height=100)
result.config(font=("Courier", 20))

#####################
## LANGUAGE CHOICE ##
#####################
# VIỆT
def on_click_languageV():
    # Cập nhật thành nội dung Tiếng Việt dưới đây
    global nbc_default, preTrain_filePath_default, test_filePath_default, \
        aftTrain_fileName_default, tokenize_clean_default
    nbc_default = nbc_E
    preTrain_filePath_default = '.\\..\\data\\processed\\before_train.pickle'
    test_filePath_default = '.\\..\\data\\processed\\test.pickle'
    aftTrain_fileName_default = "after_train.pickle"
    tokenize_clean_default = tokenize_clean_sentence

radioButton_V = Radiobutton(mainWindow, text="Vietnamese", value="V", indicatoron=0, command=on_click_languageV)
radioButton_V.config(background="light blue", font=("Courier", 15))
radioButton_V.place(x=30, y=150, width=220, height=30)

# ANH
def on_click_languageE():
    global nbc_default, preTrain_filePath_default, test_filePath_default, \
        aftTrain_fileName_default, tokenize_clean_default
    nbc_default = nbc_E
    preTrain_filePath_default = '.\\..\\data\\processed\\before_train.pickle'
    test_filePath_default = '.\\..\\data\\processed\\test.pickle'
    aftTrain_fileName_default = "after_train.pickle"
    tokenize_clean_default = tokenize_clean_sentence

radioButton_E = Radiobutton(mainWindow, text="English", value="E", indicatoron=0, command=on_click_languageE)
radioButton_E.config(background="light blue", font=("Courier", 15))
radioButton_E.place(x=250, y=150, width=220, height=30)

#####################
## CLASSIFY BUTTON ##
#####################
def on_click_classify():
    global nbc_default
    result.config(text="")
    sentence = sentenceInput.get("1.0", "end-1c")
    if sentence != "":
        result.config(text=nbc_default.classify(tokenize_clean_sentence(sentence)))

button_classify = Button(mainWindow, text="Classify",command=on_click_classify)
button_classify.place(x=30, y=180, width=440, height=30)
button_classify.config(background="light green", font=("Courier", 15))

#######################
## TRAIN TEST BUTTON ##
#######################
def on_click_train():
    global nbc_default, preTrain_filePath_default, aftTrain_fileName_default
    result.config(text="Training...")
    with open(preTrain_filePath_default, 'rb') as preTrain:
        preTrain_data = pickle.load(preTrain)

    nbc_default = NaiBay()
    nbc_default.train(preTrain_data)
    nbc_default.dumpPickleSelf(aftTrain_fileName_default)
    
    result.config(text="Trained!")

button_train = Button(mainWindow, text="Train",command=lambda:on_click_train())
button_train.place(x=30, y=350, width=220)
button_train.config(background="orange", font=("Courier", 15))

def on_click_test():
    global nbc_default, test_filePath_default
    result.config(text="Testing...")
    with open(test_filePath_default, 'rb') as test:
        test_data = pickle.load(test)

    nbc_default.test(test_data)
    result.config(text="Tested!")

button_test = Button(mainWindow, text="Test",command=on_click_test)
button_test.place(x=250, y=350, width=220)
button_test.config(background="orange", font=("Courier", 15))

mainWindow.mainloop()