# Giải nghĩa từ viết tắt, tiếng lóng
def expand_token(token):
    if token == 'u':
        return 'you'
    if token == 'r':
        return 'are'
    if token == 'some1':
        return 'someone'
    if token == 'yrs':
        return 'years'
    if token == 'hrs':
        return 'hours'
    if token == 'mins':
        return 'minutes'
    if token == 'secs':
        return 'seconds'
    if token == 'pls' or token == 'plz':
        return 'please'
    if token == '2morow':
        return 'tomorrow'
    if token == '2day':
        return 'today'
    if token == '4got' or token == '4gotten':
        return 'forget'
    if token == 'amp' or token == 'quot' or token == 'lt' or token == 'gt' or token == '½25':
        return ''
    return token

# Đưa về dạng chuẩn của từ
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

def lemmatize_token(token):
    # Chia từ thành loại: Danh từ, Động từ, Tính/trạng từ
    token = [token]
    token = pos_tag(token)

    if token[0][1].startswith("NN"):
        pos = 'n'
    elif token[0][1].startswith("VB"):
        pos = 'v'
    else:
        pos = 'a'

    # Chuẩn hóa từ
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(token[0][0], pos)

# Làm sạch từ
import re, string
from nltk.corpus import stopwords
STOP_WORDS = stopwords.words('english')

def clean_tokens(tweet_tokens):
    cleaned_tokens = []

    for token in tweet_tokens:
        # Loại bỏ các chuỗi ký tự không phải là một từ
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)",'', token)
        
        if len(token) > 0:
            # Giải nghĩa từ
            token = expand_token(token.lower())

            # Chuẩn hóa từ
            token = lemmatize_token(token)

            if token not in string.punctuation and token not in STOP_WORDS:
                cleaned_tokens.append(token)

    return cleaned_tokens

def list_to_dict_tokens(cleaned_tokens):
        newDict = dict()

        for token in cleaned_tokens:
            if token in newDict:
                newDict[token] += 1
            else:
                newDict[token] = 1

        return newDict

# Token hoá và làm sạch câu
from nltk import word_tokenize

def tokenize_clean_sentence(sentence):
    tokens = word_tokenize(sentence)
    return list_to_dict_tokens(clean_tokens(tokens))
