#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import sys




df = pd.read_csv(sys.argv[1])
print (df)




import matplotlib.pyplot as plt

x = df.x
y = df.y
plt.scatter(x,y)
plt.title('Regrex Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


plt.savefig('Scatterplot_Python.png')


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





plt.scatter(x,y)
plt.plot(X,y_predict)
plt.title('Regrex Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()






plt.scatter(x, y, color='black')



plt.show()








plt.savefig('Linearplot_Python.png')




