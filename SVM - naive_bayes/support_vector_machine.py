#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Importing the dataset
dataset = pd.read_csv('pulsar_stars.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset.columns = ['mean_profile', 'std_profile', 'kurtosis_profile', 'skewness_profile', 'mean_dmsnr',
               'std_dmsnr', 'kurtosis_dmsnr', 'skewness_dmsnr', 'target']

#Splitting into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the model
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear' , random_state = 0)
classifier.fit(X_train, y_train)


#Predicting the test result
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

#Visualising the  dataset 
plt.figure(figsize=(15,7))
vis1 = sns.countplot(dataset['target'], palette='OrRd')
plt.title('Distribution of target', fontsize=15)
plt.xlabel('Target', fontsize=13)
plt.ylabel('Count', fontsize=13)

for p in vis1.patches:
    vis1.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=13)