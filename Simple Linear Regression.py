# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:49:30 2023

@author: dell
"""

#simple linear regression#
#2) Salary_hike -> Build a prediction model for Salary_hike
#Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

#importing the data #

import numpy as np
import pandas as pd
df = pd.read_csv("D:\\data science python\\Salary_Data.csv")
df
df['Salary'].isnull().sum()

#histogram construction#

df['Salary'].hist()
df['Salary'].skew()
df['Salary'].kurt()
#constructing the boxplots#
df.boxplot(column = 'Salary',vert = False)
#scatter plot#
import matplotlib.pyplot as plt
plt.show()
df2 = df[df.columns[[0]]]
df2
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df2.iloc[:,0] = LE.fit_transform(df2.iloc[:,0])
df2.head()

#building Linaer Regression model#
#Defining X and Y variables and applying the transformations on X variable #

Y =  df['Salary']
X =  df[['YearsExperience']]
X[['YearsExperiencedSquared']] = X[['YearsExperience']]**2

X[['SquareRootYearsExperience']] = np.sqrt(X[['YearsExperience']])

X[['LogYearsExperience']] = np.log(X[['YearsExperience']])

X[['InverseSquareRootYearsExperience']] = 1 / np.sqrt(X[['YearsExperience']])

#  splitting the data into train and test data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,random_state=5)
X_train
X_test
Y_train
Y_test

#  fitting the model  #

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)
LR.intercept_#B0
LR.coef_#B1
Y_pred = LR.predict(X)
Y_pred

# calculating the metrics #

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y ,Y_pred)
print("Mean Squarred Error:",mse.round(3))
print("Root Mean Squarred Error:",np.sqrt(mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)
(r2*100).round(3)


# plotting #

import matplotlib.pyplot as plt
plt.scatter(x = 1 / np.sqrt(X['YearsExperience']),y =df["Salary"] )
plt.scatter(x = 1 / np.sqrt(X['YearsExperience']),y =Y_pred ,color = 'red' )
plt.plot(1 / np.sqrt(X['YearsExperience']),Y_pred,color='Black')    #x value is automatically taken by python###
plt.show()

# here in the above scenario, we applied a transformation on x variable and that transformation is 
#X['InverseSquareRootYearsExperience'] = 1 / np.sqrt(X['YearsExperience'])
#so after taking this transformed X value along with our target variable provides less mean squared error along with good R2 score.

















# 2 question 
# importing the data #
import pandas as pd
import numpy as np
data = pd.read_csv("D:\\data science python\\delivery_time.csv")
data
data.isnull().sum()
#no null values#

data['Delivery Time'].hist()
data['Delivery Time'].skew()#0.3523900822831107
data['Delivery Time'].kurt()#0.31795982942685397
data['Sorting Time'].hist()
data['Sorting Time'].skew()#0.047115474210530174
data['Sorting Time'].kurt()#-1.14845514534878
import matplotlib.pyplot as plt
plt.show()
plt.scatter(x = data['Delivery Time'] , y = data['Sorting Time'],color = 'red')
data.corr()
#0.825997 strongly positive relationship#

# defining X and Y variables and applying some transformations on X variable.

data.dropna(subset=['Sorting Time'], inplace=True)
Y = data['Delivery Time']

"""X =  data[['Sorting Time']]"""

X[['Sorting Time']] = X[['Sorting Time']]**2

"""X[['Sorting Time']] = np.sqrt(X[['Sorting Time']])"""

"""X[['Sorting Time']]  = np.log(X[['Sorting Time']])"""

X[['Sorting Time']] = 1 / np.sqrt(X[['Sorting Time']])


#  splitting the data into train and test data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,random_state=5)
X_train
X_test
Y_train
Y_test

# fitting the model #

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)
LR.intercept_#B0 
LR.coef_#B1 
Y_pred = LR.predict(X)
Y_pred

   # how to fix the accuracy scores of training and testing data#
    #validation set approach#
training_error = []
test_error = []
for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.75,random_state = i)
    X_train.shape
    X_test.shape
    LR.fit(X_train,Y_train)
    Y_pred_train = LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
print(training_error)
print(test_error)
print("average training error :",np.mean( training_error).round(3))   
print("average testing error :",np.mean( test_error).round(3))



# Calculating metrics #

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y ,Y_pred)
print("Mean Squarred Error:",mse.round(3))
print("Root Mean Squarred Error:",np.sqrt(mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)
(r2*100).round(3)


#plotting #import matplotlib.pyplot as plt
plt.scatter(x =1 / np.sqrt(X['Sorting Time']),y =data["Delivery Time"] )
plt.scatter(x = 1 / np.sqrt(X['Sorting Time']),y =Y_pred ,color = 'red' )
plt.plot(1 / np.sqrt(X['Sorting Time']),Y_pred,color='Black')#x value is automatically taken by python###
plt.show()


# based on the above results obtained the above model is poor model#...






























































