#Polynomial regression

#importing the libraries
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


#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)

#Training the model on the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_poly = poly_reg.fit_transform(X_train)
regressor.fit(X_poly,y_train)

#Predicting the results
y_pred = regressor.predict(poly_reg.transform(X_test))
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
     
#Evaluating the model performance'
from sklearn.metrics import r2_score
print("Performance of Polynomial regression model : ", r2_score(y_test, y_pred))
      
#Creating an array with serail numbers of values in predicted and the test set
a = np.arange(len(y_test))

#Plotting the graph
plt.plot(a,y_test , color = 'blue', label = 'original values')
plt.plot(a,y_pred , color = 'red', label = 'predicted values')
plt.legend()
plt.title('Polynomial regression model \n Model performance(r squared value) : %f '%r2_score(y_test, y_pred))
plt.show()  