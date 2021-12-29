import pickle
from tkinter import *
import tkinter
from tkinter.scrolledtext import ScrolledText
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
import seaborn as sns

from NaiveBayes_Logarit import NaiBay_L
# import làm sạch TV ở đây
# ...
from tokenize_clean_data_E import tokenize_clean_sentence
from tokenize_clean_data_V import text_preprocess

nbc_V = NaiBay_L()
nbc_V.loadPickleSelf("after_train_V_3.pickle")
nbc_E = NaiBay_L()
nbc_E.loadPickleSelf("after_train_E_3.pickle")

nbc_default = nbc_V
preTrain_filePath_default = '.\\..\\data\\processed\\before_train_V_3.pickle'
test_filePath_default = '.\\..\\data\\processed\\test_V_3.pickle'
aftTrain_fileName_default = "after_train_V_3.pickle"
tokenize_clean_default = text_preprocess

############
## WINDOW ##
############
mainWindow = Tk()
mainWindow.title("Sentiment Classifier - Nhom 20 - Nhap mon AI")
mainWindow.geometry("500x500")

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
result.place(x=150, y=380, width=200, height=50)
result.config(font=("Courier", 20))

img = Label(mainWindow)
img_NULL = ImageTk.PhotoImage(Image.open('..\\materials\\NULL.png').resize((150, 150), Image.ANTIALIAS))
img.place(x=175, y=230, width=150, height=150)

#####################
## LANGUAGE CHOICE ##
#####################
# VIỆT
def on_click_languageV():
    # Cập nhật thành nội dung Tiếng Việt dưới đây
    global nbc_default, preTrain_filePath_default, test_filePath_default, \
        aftTrain_fileName_default, tokenize_clean_default
    nbc_default = nbc_V
    preTrain_filePath_default = '.\\..\\data\\processed\\before_train_V_3.pickle'
    test_filePath_default = '.\\..\\data\\processed\\test_V_3.pickle'
    aftTrain_fileName_default = "after_train_V_3.pickle"
    tokenize_clean_default = text_preprocess

radioButton_V = Radiobutton(mainWindow, text="Vietnamese", value="V", indicatoron=0, command=on_click_languageV, relief=SUNKEN)
radioButton_V.config(background="light blue", font=("Courier", 15))
radioButton_V.place(x=30, y=150, width=220, height=30)

# ANH
def on_click_languageE():
    global nbc_default, preTrain_filePath_default, test_filePath_default, \
        aftTrain_fileName_default, tokenize_clean_default
    nbc_default = nbc_E
    preTrain_filePath_default = '.\\..\\data\\processed\\before_train_E_3.pickle'
    test_filePath_default = '.\\..\\data\\processed\\test_E_3.pickle'
    aftTrain_fileName_default = "after_train_E_3.pickle"
    tokenize_clean_default = tokenize_clean_sentence

radioButton_E = Radiobutton(mainWindow, text="English", value="E", indicatoron=0, command=on_click_languageE)
radioButton_E.config(background="light blue", font=("Courier", 15))
radioButton_E.place(x=250, y=150, width=220, height=30)

#####################
## CLASSIFY BUTTON ##
#####################
img_NEG = ImageTk.PhotoImage(Image.open('..\\materials\\NEG.png').resize((150, 150), Image.ANTIALIAS))
img_NEU = ImageTk.PhotoImage(Image.open('..\\materials\\NEU.png').resize((150, 150), Image.ANTIALIAS))
img_POS = ImageTk.PhotoImage(Image.open('..\\materials\\POS.png').resize((150, 150), Image.ANTIALIAS))

def on_click_classify():
    global nbc_default, img
    result.config(text="")
    sentence = sentenceInput.get("1.0", "end-1c")
    if sentence != "":
        print(tokenize_clean_default(sentence))
        text = nbc_default.classify(tokenize_clean_default(sentence))
        if text == 'NEG':
            img.config(image=img_NEG)
            result.config(text='NEGATIVE', fg="red")
        elif text == 'POS':
            img.config(image=img_POS)
            result.config(text='POSITIVE', fg="green")
        else:
            img.config(image=img_NEU)
            result.config(text='NEUTRAL', fg="blue")

button_classify = Button(mainWindow, text="Classify",command=on_click_classify)
button_classify.place(x=30, y=180, width=440, height=30)
button_classify.config(background="light green", font=("Courier", 15))

#######################
## TRAIN TEST BUTTON ##
#######################
def on_click_train():
    global nbc_default, preTrain_filePath_default, aftTrain_fileName_default
    result.config(text="Training...")
    img.config(image=img_NULL)

    with open(preTrain_filePath_default, 'rb') as preTrain:
        preTrain_data = pickle.load(preTrain)

    nbc_default = NaiBay_L()
    nbc_default.train(preTrain_data)
    nbc_default.dumpPickleSelf(aftTrain_fileName_default)
    
    result.config(text="Trained!", fg="orange")

button_train = Button(mainWindow, text="Train",command=lambda:on_click_train())
button_train.place(x=30, y=450, width=220)
button_train.config(background="orange", font=("Courier", 15))

def on_click_test():
    global nbc_default, test_filePath_default
    result.config(text="Testing...")
    img.config(image=img_NULL)

    with open(test_filePath_default, 'rb') as test:
        test_data = pickle.load(test)

    testResult = nbc_default.test(test_data)
    
    ax = sns.heatmap(testResult[0], annot=True, cmap='Blues', fmt='g')
    ax.set_title(testResult[1])
    ax.set_xlabel('Nhãn dự đoán')
    ax.set_ylabel('Nhãn thực')
    ax.xaxis.set_ticklabels(testResult[2])
    ax.yaxis.set_ticklabels(testResult[2])
    plt.show()

    result.config(text="Tested!", fg="orange")

button_test = Button(mainWindow, text="Test",command=on_click_test)
button_test.place(x=250, y=450, width=220)
button_test.config(background="orange", font=("Courier", 15))

mainWindow.mainloop()