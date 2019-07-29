
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

print('ji')
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
#print(x)
#print(y)


# Taking care of missing data

datasetmiss = pd.read_csv('Data_miss.csv')
xm = datasetmiss.iloc[:,:-1].values
ym = datasetmiss.iloc[:,3].values 


from  sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(xm[:,1:3])
xm[:,1:3] = imputer.transform(xm[:,1:3])

#encoding categorial data

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x = LabelEncoder()
xm[:,0] = labelencoder_x.fit_transform(xm[:,0])

onehotencoder = OneHotEncoder(categorical_features=[0])
xm = onehotencoder.fit_transform(xm).toarray()

labelencoder_y = LabelEncoder()
ym = labelencoder_x.fit_transform(ym)

#splitting the data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(xm,ym,test_size=0.2,random_state =0)


#FEATURE SCALING

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
