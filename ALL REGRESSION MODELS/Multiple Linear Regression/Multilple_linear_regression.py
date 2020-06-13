#Multiple linear regression

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing the dataset
dataset = pd.read_csv('RealEstate_Data.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Visualising the whole dataset values
a = np.arange(len(y))
plt.bar(a,y , color = 'darkgreen' )
plt.title('House prices per unit area determined by various factors' )
plt.show()

#Splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X , y , test_size = 0.20, random_state = 0)
 
#Training the model on the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1) , y_test.reshape(len(y_test),1) ),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print("Performance of Multiple linear regression model : ",r2_score(y_test, y_pred))

#Creating an array with serail numbers of values in predicted and the test set
a = np.arange(len(y_test))

#Plotting the graph
plt.plot(a,y_test , color = 'blue', label = 'original values')
plt.plot(a,y_pred , color = 'red', label = 'predicted values')
plt.legend()
plt.title('Multiple Linear Regression model \n Model performance(r squared value) : %f '%r2_score(y_test, y_pred))
plt.show()