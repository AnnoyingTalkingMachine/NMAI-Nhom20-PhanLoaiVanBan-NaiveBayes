import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    cmap=plt.get_cmap('Blues')
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

    

    plt.show()

def to_df_confusion(y_actu, y_pred):
    actual = pd.Series(y_actu, name='Actual')
    predicted = pd.Series(y_pred, name='Predicted')
    return pd.crosstab(actual, predicted)
