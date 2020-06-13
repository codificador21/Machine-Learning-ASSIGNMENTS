#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 

Test = pd.read_csv('Test.csv')
Train = pd.read_csv('Train.csv')

X_train = Train.iloc[:,:-1].values
y_train = Train.iloc[:,-1].values
X_test = Test

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

import statsmodels.api as sm

X_opt = np.array(X_train[:, [0, 1, 2, 3, 4]], dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
print(regressor_OLS.summary())

#Creating an array with serail numbers of values in predicted and the test set
s_no = dataset.iloc[:len(X_test),0].values

#Plotting the graph
plt.plot(s_no,y_test , color = 'blue', label = 'original values')
plt.plot(s_no,y_pred , color = 'red', label = 'predicted values')
plt.legend()
plt.show()

"""THE P-VALUES FOR ALL VARIABLES ARE ZERO,
HENCE ALL THE INDEPENDENT VARIABLES ARE SIGNIFICANT,
THEREFORE WE WILL NOT ELIMINATE ANY OF THE VARIABLES !!!"""