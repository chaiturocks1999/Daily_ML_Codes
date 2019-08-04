# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

from  sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]])))) 

plt.scatter(x,y,color = 'red')
plt.plot(x,regressor.predict(x),color = 'blue')
plt.title('truth or bluff(SVR )')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color = 'red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
#plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color = 'blue')
plt.title('truth or bluff(linear regression )')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
