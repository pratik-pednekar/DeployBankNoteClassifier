#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 23:15:45 2021

@author: pratik
"""

#Library imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

#Obtaining the data
df=pd.read_csv('BankNote_Authentication.csv')
X=df[['variance','skewness','curtosis','entropy']]
y=df['class']

#Processing data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=0)

#Generating the model
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

#Scoring
y_pred=classifier.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)*100
print(round(accuracy,2))

#Saving model file
pickle_out=open("BankNoteClassifier_RFM.pkl",'wb')
pickle.dump(classifier,pickle_out)
pickle_out.close

#random prediction
print(classifier.predict([[2,3,4,1]]))