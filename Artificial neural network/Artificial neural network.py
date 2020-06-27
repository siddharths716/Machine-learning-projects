import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[: , 3:13].values
Y = dataset.iloc[:, 13].values

#encoding categorical data
#transforming 2 type categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X[ : , 2] = label_encoder.fit_transform(X[ : , 2])

#transforming 3 or more type categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder' , OneHotEncoder() , [1])] , remainder = 'passthrough' )
X = np.array(ct.fit_transform(X))
X = X[:, 1:]  #avoiding the dummy variable trap

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#MAKING THE ARTIFICIAL NEURAL NETWORK

#importing the keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising the ANN
classifier = Sequential()

#adding the input layer and the first hidden layer
classifier.add( Dense(output_dim = 6 , init = 'uniform' , activation = 'relu' , input_dim = 11))

#adding the second hidden layer
classifier.add( Dense(output_dim = 6 , init = 'uniform' , activation = 'relu' ))

#adding the output layer
classifier.add( Dense(output_dim = 1 , init = 'uniform' , activation = 'sigmoid' ))

#compiling the artificial neural network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#ftting the ANN to the training set
classifier.fit(x_train , y_train, batch_size = 10, epochs = 100 )

#predicting the test results
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)

