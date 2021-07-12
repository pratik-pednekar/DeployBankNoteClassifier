#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 23:14:52 2021

@author: pratik
"""

#Library imports
import uvicorn                   #Need to install this library
from fastapi import FastAPI        #Need to install this library
from banknotes import BankNote
import numpy as np
import pickle                    #Need to install this library
import pandas as pd

#Instantiating model and API
app = FastAPI()
pickle_in = open('BankNoteClassifier_RFM.pkl','rb')
classifier = pickle.load(pickle_in)

#Index route, opens automatically on port 8000 [http://127.0.0.1:8000]
@app.get('/')
def index():
    return {"message" : "Welcome to the BankNote Classifier App by Pratik"}

#Testing intake of parameter [http://127.0.0.1:8000/*args]
#@app.get('/{name}')
#def new_name(name:str):
#    return {"message" : f'Hello {name}'}

#Making and returning a prediction from the ML model
@app.post('/predict')
def prediction(data:BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    
    predicted=classifier.predict([[variance,skewness,curtosis,entropy]])
    
    if predicted<0.5:
        pred="The note is a REAL Bank note"
    else:
        pred="The note is FAKE"
    return {
        'prediction':pred
        }

#Deployment
if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
    
#Command to launch app:
#uvicorn app:app --reload


    