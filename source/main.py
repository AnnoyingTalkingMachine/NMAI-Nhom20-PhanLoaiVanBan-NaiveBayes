from NaiveBayes import NaiBay
nbc = NaiBay()
nbc.loadPickleSelf()

from tokenize_clean_data import tokenize_clean_sentence
print(nbc.classify(tokenize_clean_sentence('Nice, yo mate sunny innit')))