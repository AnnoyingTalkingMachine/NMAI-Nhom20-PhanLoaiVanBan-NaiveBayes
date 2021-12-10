# Đọc tập train
import pickle

with open('.\\..\\data\\processed\\after_train.pickle', 'rb') as after_train:
    trained_data = pickle.load(after_train)

print(trained_data)