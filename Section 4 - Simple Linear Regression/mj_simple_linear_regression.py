import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#print('ji')

# Taking care of missing data

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#splitting the data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state =0)


#linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('salary vs experience (Training set )')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('salary vs experience (Training set )')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()