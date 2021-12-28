# Tiền xử lý dữ liệu -> đưa về dạng từ điển#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import regex as re
 

#  Đồng bộ unicode # 

def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
 
dicchar = loaddicchar()
 
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def convert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)


# Tokenize #
from pyvi import ViTokenizer
from underthesea import word_tokenize

#Remove HTML code#
def remove_html(txt):
    return re.sub(r'<[^>]*>', '', txt)

#Clean token#
def clean_token(token):
    if(token == "k" or token == "ko" or token == "khong" or token == "kg"): return "không"
    if(token == "dc" or token == "đc"): return "được"
    if(token == "sz"): return "size"
    if(token == "siu"): return "siêu"
    if(token == "cx" or token == "cug"): return "cũng"
    if(token == "bt"): return "bình_thường"
    if(token == "iu"): return "yêu"
    if(token == 'i'): return "y"
    if(token == "oke" or token == "okie" or token =="okz"): return "ok"
    if(token == "xau"): return "xấu"
    if(token == "tks"): return "thanks"
    if(token == "dep"): return "đẹp"
    if(token == "tam"): return "tạm"
    if(token == "loi"): return "lỗi"
    if(token == "hai long"): return "hài_lòng"
    if(token == "tot"): return "tốt"
    if(token == "lam"): return "lắm"
    if(token == "rat"): return "rất"

    return token

#Remove stopwords#
words =  pd.read_csv('../data/VN/vietnamese-stopwords.txt', header = None)

stop_word = set(words.values.ravel())
final_stop_word = set()

for i in stop_word:
    a = i.replace(" ","_")
    final_stop_word.add(a)

def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in final_stop_word:
            words.append(word)
    return ' '.join(words)

#xóa bỏ các kí tự thừa#
def remove_redundant_character(document):
    
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]',' ',document)
    document = re.sub("[0-9].kg", "", document) #remove cân nặng
    document = re.sub("[0-9].k", "", document) #remove giá tiền
    document = re.sub("[0-9]", "", document) #remove number
    cha = ('a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','u','z','r','s')
    for i in cha:
        document = re.sub(f"{i}{i}*",f"{i}",document)
    return document

#chuyển hóa về dạng từ điển#
def list_to_dict_tokens(cleaned_tokens):
        newDict = dict()

        for token in cleaned_tokens:
            if token in newDict:
                newDict[token] += 1
            else:
                newDict[token] = 1

        return newDict


#Summary: text preprocess#
def text_preprocess(document):
    # xóa html code
    document = remove_html(document)

    # chuẩn hóa unicode
    document = convert_unicode(document)

    # đưa về lower
    document = document.lower()

    # xóa các ký tự không cần thiết
    document = remove_redundant_character(document)

    # xóa khoảng trắng thừa
    document = re.sub(r'\s+', ' ', document).strip()

    # tokenize
    document = ViTokenizer.tokenize(document)
    document = word_tokenize(document)
    
    # clean tokens
    for i in range(0, len(document)):
        document[i] = clean_token(document[i])
    document = " ".join(str(e) for e in document)

    #remove stopwords
    document = remove_stopwords(document)

    # đưa về dạng từ điển
    document = document.split(' ')
    document = list_to_dict_tokens(document)
    
    return document