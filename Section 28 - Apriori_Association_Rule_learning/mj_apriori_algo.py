# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:55:57 2019

@author: Mahender jakhar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
# Data Preprocessing
# Here we use header = "None" because we dont want the columm name in our dataset  
dataset = pd.read_csv('Market_Basket_Optimisation.csv')

#here we want list of lists in the parameter of apriori due to which we convert our data set in the 
#list of lists  for which we use str to convert it into strings 

transactions = []
for i in range(0, 7501):
        transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
        
    

# Training Apriori on the dataset
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)