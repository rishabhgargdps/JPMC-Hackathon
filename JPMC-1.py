#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint


# In[ ]:


df = pd.read_csv("/home/rishabhgarg/Documents/JPMC-1/input1.csv")
for i in range(1,101):
    df['ratio'+str(i)] = df['F_'+str(i)]/df['F_'+str(i)].shift()-1
df = df.iloc[1:]
df.drop(columns = {'Day', 'Month', 'Year'}, inplace = True)
temp = df[df.columns[100:200]]


# In[ ]:


weights = np.random.random(100).ravel()
constraint = LinearConstraint(np.ones(10), lb=0, ub=1)
cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
bnds = tuple((0,1) for x in weights)
def getSharpe(weights, temp):
    final_matrix = temp*weights
    res = final_matrix.sum(axis = 1, skipna = True)
    return -res


# In[ ]:


res = minimize(getSharpe, x0 = weights, args = (temp), method='SLSQP', bounds=bnds ,constraints=cons)

