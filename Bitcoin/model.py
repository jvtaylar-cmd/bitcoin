#import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('bitcoin.csv')
dataset.head()

X = dataset.loc[:,['Open']].values
y = dataset.loc[:,['Close']].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Prediction on the Closing Price using Opening Price
open_price = 155
close = regressor.predict([[open_price]])
print(close)

import pickle
pickle.dump(regressor,open('crypto_model.pkl','wb')) #saving our model in .pkl file

