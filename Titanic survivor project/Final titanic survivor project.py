from tkinter import *
root = Tk()
root.title("TITANIC SURVIVER")
root.geometry("800x800")
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt





def classify():
    train = pd.read_csv("train.csv")
    train_x = train.iloc[: , [2,4,5,6,7,9,11]]
    #FOR TEST SET
    test = pd.read_csv("test.csv")
    test_x = test.iloc[: , [1,3,4,5,6,8,10]]
    
    a = int(entry0.get())
    b = str(entry1.get())
    c = int(entry2.get())
    d = int(entry3.get())
    e = int(entry4.get())
    f = entry5.get()
    g = str(entry6.get())
    combined = pd.concat([train_x,test_x])
    combined = combined.append({'Pclass': a , "Sex": b  , "Age": c , "SibSp": d  , "Parch": e , "Fare" : f , "Embarked": g } , ignore_index = True  )
    
    X = combined.values
    train_y = train.iloc[ : , 1:2].values
    #adding the new row to the data
    
    
    
    #taking care of missing values training set
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
    X[:, 2:3] = imputer.fit_transform( X[:, 2:3])
    imputer1 = SimpleImputer(missing_values = np.nan , strategy = 'most_frequent')
    X[:, 6:7] = imputer1.fit_transform( X[:, 6:7])
    imputer = SimpleImputer(missing_values = np.nan , strategy = 'median')
    X[:, 5:6] = imputer.fit_transform( X[:, 5:6])
    
    #taking care of 2 categorical values
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    X[ : , 1] = label_encoder.fit_transform(X[ : , 1])
    
    #transforming 3 or more type categorical data for training set
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers = [('encoder' , OneHotEncoder() , [0])] , remainder = 'passthrough' )
    X = np.array(ct.fit_transform(X))
    X = X[:, 1:]  #avoiding the dummy variable trap
    
    ct1 = ColumnTransformer(transformers = [('encoder' , OneHotEncoder() , [4])] , remainder = 'passthrough' )
    X = np.array(ct1.fit_transform(X))
    X = X[:, 1:]  #avoiding the dummy variable trap
    
    ct2 = ColumnTransformer(transformers = [('encoder' , OneHotEncoder() , [10])] , remainder = 'passthrough' )
    X = np.array(ct2.fit_transform(X))
    X = X[:, 1:]  #avoiding the dummy variable trap
    
    ct3 = ColumnTransformer(transformers = [('encoder' , OneHotEncoder() , [18])] , remainder = 'passthrough' )
    X = np.array(ct3.fit_transform(X))
    X = X[:, 1:]  #avoiding the dummy variable trap
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    pred = X[ -1 , : ]
    
    
    #splitting into training and testing
    train_x = X[:891 , : ]
    test_x = X[892: , :]
    
    # Training the Random Forest Classification model on the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
    classifier.fit(train_x, train_y)
    
    y_pred = classifier.predict([pred]) 
    
    entry0.delete( 0 , END)
    entry1.delete( 0 , END)
    entry2.delete( 0 , END)
    entry3.delete( 0 , END)
    entry4.delete( 0 , END)
    entry5.delete( 0 , END)
    entry6.delete( 0 , END)
    
    if y_pred == [0]:
        dis = Label(root , text = "Sorry you would die" )
        dis.grid(row = 8 , column = 0 , padx= 30)
    elif y_pred==[1]:
        dis = Label(root , text = "You would survive" )
        dis.grid(row = 8 , column = 0 , padx= 30)
    #dis = Label(root , text = y_pred )
    #dis.grid(row = 9 , column = 0 , padx= 30)
    
    
    
    
    
 #k_fold cross validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = classifier, X= train_x , y= train_y , cv=10)
# accuracies.mean()
mylabel0 = Label(root , text = 'Your class' )
mylabel0.grid(row = 0 , column = 0 , padx = 30 )
entry0 = Entry(root)
entry0.grid(row = 0 , column = 1 , padx = 40   )

mylabel1 = Label(root , text = 'Your gender' )
mylabel1.grid(row = 1 , column = 0 , padx = 30 )
entry1 = Entry(root)
entry1.grid(row = 1 , column = 1 , padx = 40   )

mylabel2 = Label(root , text = 'Your age' )
mylabel2.grid(row = 2 , column = 0 , padx = 30 )
entry2 = Entry(root)
entry2.grid(row = 2, column = 1 , padx = 40   )

mylabel3 = Label(root , text = 'Siblings you would take to the ship' )
mylabel3.grid(row = 3 , column = 0 , padx = 30 )
entry3 = Entry(root)
entry3.grid(row = 3 , column = 1 , padx = 40   )

mylabel4 = Label(root , text = 'Parents you would take' )
mylabel4.grid(row = 4 , column = 0 , padx = 30 )
entry4 = Entry(root)
entry4.grid(row = 4 , column = 1 , padx = 40   )

mylabel5 = Label(root , text = 'Your fare' )
mylabel5.grid(row = 5 , column = 0 , padx = 30 )
entry5 = Entry(root)
entry5.grid(row = 5 , column = 1 , padx = 40   )

mylabel6 = Label(root , text = 'Port of embarkment' )
mylabel6.grid(row = 6 , column = 0 , padx = 30 )
entry6 = Entry(root)
entry6.grid(row = 6 , column = 1, padx = 40  )

mybutton = Button( root , text = 'submit' , command = classify)
mybutton.grid(row = 7 , column = 0)
root.mainloop()