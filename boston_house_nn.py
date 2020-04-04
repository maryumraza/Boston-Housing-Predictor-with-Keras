# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:50:20 2020

@author: uni tech
"""



from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# Initializing train and test datasets
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()



# Scaling the inputs
X_train_fit = StandardScaler().fit(X_train)
X_train = X_train_fit.transform(X_train)

X_test = X_train_fit.transform(X_test)



# Defining model and adding layers
model = Sequential()

model.add(Dense(128, input_shape= ( X_train.shape[1], ), activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dense(1))




# Comilation
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])



# Model training
model.fit(X_train, y_train, epochs=25)



# Calculating the accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(accuracy)
    

   

























