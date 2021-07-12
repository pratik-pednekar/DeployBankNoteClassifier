#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 23:35:54 2021

@author: pratik
"""

#Importing libraries
from pydantic import BaseModel

#Class that describes banknote ML models inputs

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float
    