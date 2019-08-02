"""
Created on Thu Aug  1 20:18:45 2019

@author: Mahender jakhar
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd



dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Polynomial regression 

from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)


plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.title('truth or bluff(linear regression )')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color = 'blue')
#plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color = 'blue')
plt.title('truth or bluff(linear regression )')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()