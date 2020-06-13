import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Linear_X_Test = pd.read_csv('Linear_X_Test.csv')
Linear_X_Train = pd.read_csv('Linear_X_Train.csv')
Linear_Y_Train = pd.read_csv('Linear_Y_Train.csv')

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Linear_X_Train,Linear_Y_Train)

Linear_Y_predict = regressor.predict(Linear_X_Test)

plt.scatter(Linear_X_Train,Linear_Y_Train,color = 'red',label = 'Original values')
plt.plot(Linear_X_Test,Linear_Y_predict,color = 'blue', label = 'predicted values')
plt.legend()
plt.show()