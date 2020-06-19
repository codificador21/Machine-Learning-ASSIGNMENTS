#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
X_train = pd.read_csv('Diabetes_XTrain.csv')
X_test = pd.read_csv('Diabetes_Xtest.csv')
y_train = pd.read_csv('Diabetes_YTrain.csv')
y_train = y_train.iloc[:,:].values

#Visualising the dataset
y = y_train.reshape(len(y_train))
a = np.arange(len(y_train))
plt.bar(a,y,color = 'indigo' )
plt.title('Visualising the dataset')
plt.show()

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the model
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)

#Training the model
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier2.fit(X_train, y_train) 

#Predicting the test result
y_pred_LR = classifier1.predict(X_test)
y_pred_KNN = classifier2.predict(X_test)
print(np.concatenate((y_pred_LR.reshape(len(y_pred_LR),1), y_pred_KNN.reshape(len(y_pred_KNN),1)),1))

a = np.arange(len(X_test))

#Visualising the predicted values
plt.bar(a , y_pred_LR , color = 'red',label = 'logistic regression')
plt.bar(a , y_pred_KNN , color = 'blue' , label = 'K nearest neighbors')
plt.legend()
plt.title('Comparison of predicted values')
plt.show()