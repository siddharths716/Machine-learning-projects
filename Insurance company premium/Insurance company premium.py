# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Final+Test+Data+set.csv')
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12].values

#taking care of missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
X[:, 7:9] = imputer.fit_transform( X[:, 7:9])
imputer2 = SimpleImputer(missing_values = np.nan , strategy = 'most_frequent')
X[:, [0,1,2,4,9]] = imputer2.fit_transform( X[:, [0,1,2,4,9]])

#taking care of categorical values
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X[ : , 0] = label_encoder.fit_transform(X[ : , 0])
label_encoder1 = LabelEncoder()
X[ : , 1] = label_encoder1.fit_transform(X[ : , 1])
label_encoder3 = LabelEncoder()
X[ : , 3] = label_encoder3.fit_transform(X[ : , 3])
label_encoder4 = LabelEncoder()
X[ : , 4] = label_encoder4.fit_transform(X[ : , 4])
label_encoder9 = LabelEncoder()
X[ : , 9] = label_encoder9.fit_transform(X[ : , 9])

#transforming 3 or more type categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder' , OneHotEncoder() , [2])] , remainder = 'passthrough' )
X = np.array(ct.fit_transform(X))
X = X[:, 1:]  #avoiding the dummy variable trap

#transforming 3 or more type categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct1 = ColumnTransformer(transformers = [('encoder' , OneHotEncoder() , [12])] , remainder = 'passthrough' )
X = np.array(ct1.fit_transform(X))
X = X[:, 1:]  #avoiding the dummy variable trap


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

