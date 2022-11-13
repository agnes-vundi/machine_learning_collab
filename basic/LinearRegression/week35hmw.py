import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# read data from panda
df = pd.read_csv('startup.csv')
X = df.iloc[:,:-1]                 #take everything except last one 
y = df.iloc[:,[-1]]                #take last one

#Splitting the dataset into the Training set(80%) and Test set(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#train the Multiple linear regression model using the training sets
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)

#predict the test set results
y_pred = regr.predict(X_test)

#calculate and analyze regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print ('\nTestdata metrics:')
print("Mean Absolute Error is :" ,mae)
print("Mean Squarred Error is :" ,mse)
print("Root Mean Squarred Error is : ",rmse)
print("R2 score of model is :" ,r2)

#Predict also the profit for a new company
new_company = np.array([[[165349.2,136897.8,471784.1]]])
reshaped_array = np.reshape(new_company,(1,3))
PrePro = regr.predict(reshaped_array)
print("The predict profit for new campany is ",PrePro)
