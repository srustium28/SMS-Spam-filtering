#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:03:18 2020

@author: almas
"""

import numpy as np
import pandas as pd

from sklearn import feature_extraction, model_selection, naive_bayes

import warnings
warnings.filterwarnings("ignore")
 
data = pd.read_csv("spam.csv")

f = feature_extraction.text.CountVectorizer(stop_words = 'english')


X = f.fit_transform(data["v2"])
print(X)
np.shape(X)

y=data["v1"].map({'spam':1,'ham':0})

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=0)
print([np.shape(X_train), np.shape(X_test)])


list_alpha = np.arange(1/100000, 20, 0.11)

for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)


from sklearn.feature_extraction import DictVectorizer
#from keras.models import load_model


mytest=("WINNER!! As a valued network customer you have been selected to receivea �900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only. ")
Y = [mytest]#mytest is a new email in string format
f = feature_extraction.text.CountVectorizer(stop_words = 'english')

f.fit(data["v2"]) # fittingf

X = f.transform(Y)
res=bayes.predict(X)
if res == 1:
    result = "spam"
    print(result)
    
else:
    result = "ham"
    print(result)

