# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:27:54 2020

@author: uni tech
"""




from keras.datasets import boston_housing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler



# Initializing train and test datasets
(X_train, y_train) , (X_test, y_test) = boston_housing.load_data()



# Scaling the inputs
X_train_fit = StandardScaler().fit(X_train)
X_train = X_train_fit.transform(X_train)


X_test = X_train_fit.transform(X_test)




# Defining regression algorithms
def linear_reg():
    clf= LinearRegression()
    return clf

    
    
def svm_reg():
    clf= SVR(kernel='rbf', degree=3, gamma='scale')
    return clf



def decision_tree():
    clf=DecisionTreeRegressor(criterion='mse',splitter='best')
    return clf



def random_forest():
    clf= RandomForestRegressor(n_estimators=5, criterion='mse')
    return clf




LINEAR_REGRESSION = linear_reg()
SUPPORT_VECTOR = svm_reg()
DECISION_TREE = decision_tree()
RANDOM_FOREST = random_forest()



models = [LINEAR_REGRESSION, SUPPORT_VECTOR, DECISION_TREE, RANDOM_FOREST ]
scores=[]

for model in models:
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    scores.append(score)        


print(scores)

 
    
    
    