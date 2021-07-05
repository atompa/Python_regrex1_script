#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv ("regrex1.csv")
print (df)


# In[5]:


import matplotlib.pyplot as plt

x = df.x
y = df.y
plt.scatter(x,y)
plt.title('Regrex Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[6]:


import numpy as np
from sklearn.linear_model import LinearRegression
X = df.x.to_numpy()
X = X.reshape(-1, 1)
y = df.y.to_numpy ()
y = y.reshape(-1,1)
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_
y_predict = reg.predict(X)


# In[7]:


plt.scatter(x,y)
plt.plot(X,y_predict)
plt.title('Regrex Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:

plt.scatter(x, y, color='black')



plt.show()




# In[ ]:




plt.savefig('Linearplot_Python.png')




