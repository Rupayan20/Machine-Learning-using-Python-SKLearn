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
