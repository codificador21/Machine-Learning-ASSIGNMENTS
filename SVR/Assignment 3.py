#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Car_Purchasing_Data.csv',encoding='latin-1')
X = dataset.iloc[:,3:-1].values  #Independent variables
y = dataset.iloc[:,-1].values    #Dependent variables

y= y.reshape(len(y),1)

#FeatureScaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

#Splitting the data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y ,test_size = 0.2 , random_state = 0 )

#Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)

y_test = sc_y.inverse_transform(y_test)


#Predicting the new result
y_pred = sc_y.inverse_transform(regressor.predict(X_test))
X_test = sc_X.inverse_transform(X_test)

#Comparing the results
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Creating an array with serail numbers of values in predicted and the test set
s_no = dataset.iloc[:len(X_test),0].values

#Plotting the graph
plt.plot(s_no,y_test , color = 'blue', label = 'original values')
plt.plot(s_no,y_pred , color = 'red', label = 'predicted values')
plt.legend()
plt.show()
