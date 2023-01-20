'''
Project: Real Estate Price Predication
Created By: Rupayan Dutta
Guided By: Mr. Shaurya Sharma
'''

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# creating a Dataframe
boston= load_boston()
df= pd.DataFrame(boston.data)

# EDA- Exploratory Data Analysis
df.head()


# adding Coloums
df.columns= boston.feature_names
df.head()

'''
Columns Information
CRIM per capita crime rate by town
ZN proportion of residential land zoned for lots over 25,000 sq.ft.
ZN proportion of residential land zoned for lots over 25,000 sq.ft.
CHAS Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX nitric oxides concentration (parts per 10 million)
RM average number of rooms per dwelling
AGE proportion of owner-occupied units built prior to 1940
DIS weighted distances to five Boston employment centres
RAD index of accessibility to radial highways
TAX full-value property-tax rate per 10,000usd
PTRATIO pupil-teacher ratio by town
B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT % lower status of the population
Adding the target column into the DataFrame
'''
df['PRICE'] = boston.target
df.head()
df.tail()
df.shape
df.columns
df.dtypes
df.nunique()
df.isnull()
df.isnull().sum()
df.describe()
df.corr

plt.figure(figsize=(10,10))
sns.heatmap(data=df.corr(),annot=True,cmap='Greens')

sns.pairplot(df,size=5)

# Plot a Boxplot
plt.figure(figsize=(50,50))
df.boxplot()

# Minimum Price
df.PRICE.min()

# Maximum Price
df.PRICE.max()

# Standard Deviation
df.PRICE.std()

Export the dataset
df.to_csv('boston_datset.csv',)
df.head()

X = np.array(df.drop('PRICE', axis=1))
y = np.array(df.PRICE)
# X = boston.data
# y = boston.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
len(X_train)
len(y_train)
len(X_test)
len(y_test)
model = LinearRegression()
model.fit(X_train,y_train)

# Intercept Value
model.intercept_

# Prediction
# Testing the model performance
model.score(X_test,y_test)

# R squared
r2_score(y_test,y_pred)

# Adjusted R squared
# MSE
mean_squared_error(y_test,y_pred)

# MAE
mean_absolute_error(y_test,y_pred)

# RMSE
np.sqrt(mean_squared_error(y_test,y_pred))

plt.scatter(y_test,y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.grid() 
plt.plot([min(y_test),max(y_test)],[min(y_pred),max(y_pred)], ,color='red')
plt.title('Actual Price vs Predicted Price')
plt.show()
