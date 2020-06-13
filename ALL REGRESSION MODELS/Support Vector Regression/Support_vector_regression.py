#Support vector regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('RealEstate_Data.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Visualising the whole dataset values
a = np.arange(len(y))
plt.bar(a,y , color = 'darkgreen' )
plt.title('House prices per unit area determined by various factors' )
plt.show()

y = y.reshape(len(y),1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X , y , test_size = 0.20, random_state = 0)

#Training the model on the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train) 

y_test = sc_y.inverse_transform(y_test)

#Predicting the rsults
y_pred = sc_y.inverse_transform(regressor.predict(X_test))
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test),1))

#Evaluating the model performance
from sklearn.metrics import r2_score
print("Performance of SVR model : ",r2_score(y_test,y_pred))

#Creating an array with serail numbers of values in predicted and the test set
a = np.arange(len(y_test))

#Plotting the graph
plt.plot(a,y_test , color = 'blue', label = 'original values')
plt.plot(a,y_pred , color = 'red', label = 'predicted values')
plt.legend()
plt.title('Support vector regression model \n Model performance(r squared value) : %f '%r2_score(y_test, y_pred))
plt.show()