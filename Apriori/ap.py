# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dataset = pd.read_csv(r"F:\Udemy\Machine Learning\Machine Learning A-Z Template Folder\Part 5 - Association Rule Learning\Section 28 - Apriori\Market_Basket_Optimisation.csv", header=None)

transaction = []
for i in range(0,7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])

#Apriori accepts in string format

from apyori import apriori
rules = apriori(transaction,min_support = 0.003, min_confidence = 0.2 , min_lift = 3, min_length = 2)

results = list(rules)