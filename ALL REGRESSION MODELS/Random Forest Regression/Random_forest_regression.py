#Random forest regression

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('RealEstate_Data.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Visualising the whole dataset values
a = np.arange(len(y))
plt.bar(a,y , color = 'darkgreen' )
plt.title('House prices per unit area determined by various factors' )
plt.show()


#Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#training the model on the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor( n_estimators = 100, random_state =0)
regressor.fit(X_train,y_train)

#Predicting the results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Evaluating the Model performance
from sklearn.metrics import r2_score
print("Performance of Random Tree Regression model : ",r2_score(y_test, y_pred))

#Creating an array with serail numbers of values in predicted and the test set
a = np.arange(len(y_test))

#Plotting the graph
plt.plot(a,y_test , color = 'blue', label = 'original values')
plt.plot(a,y_pred , color = 'red', label = 'predicted values')
plt.legend()
plt.title('Random forest regression model \n Model performance(r squared value) : %3f '%r2_score(y_test, y_pred))
plt.show()