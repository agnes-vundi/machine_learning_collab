import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read data to pandas dataframe
df = pd.read_csv('titanic.csv')

# Remove the row without Cabin class (PClass as '*' in data)
# If the row is not removed, it will be considered as a separate cabin class
df = df[df['PClass']!='*']

# Create input data (X) and output data (y)
X = df[['PClass','Age','GenderCode']] # or df.loc[:, ['PClass','Age','GenderCode']]
y = df[['Survived']]

# Split data into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save the original format data in readable format for the later testing phase
X_test_orig = X_test

# Transform categorical text values in two columns into numeric columns for each category in column
# pandas.get_dummies transforms only categorical columns, and it does not touch numerical columns
# Use drop_first to remove one column to avoid the 'dummy variable trap'
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Feature scaling - normalise (scale) all numeric data into range [0, 1]
# In this case column 'Age' must be scale to [0, 1] to prevent 'Age' column domination in the model
max_value = X_train['Age'].max()
X_train['Age'] = X_train['Age'] / max_value
X_test['Age'] = X_test['Age'] / max_value

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train.values.ravel())

# Predicting outputs with X_test as inputs
y_pred = model.predict(X_test)

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

# Create empty dataframe for the results
test_results = pd.DataFrame()

# Get original test values from X_test (values are saved in X_test_orig)
# These are the human readable original columns: 'PClass', 'Age', 'GenderCode'
test_results = X_test_orig

# Add the original and real values from y_test
test_results['Real output'] = y_test.values

# Add the predicted results as a new column to results
test_results['Predicted output'] = y_pred

# Estimate the result by confusion matrix
# TP FN
# FP TN
cm = confusion_matrix(y_test, y_pred)

# Check the accuracy of the results
accuracy = accuracy_score(y_test, y_pred)
print("accuracy is ",accuracy)

# Single input prediction, using the same scaling as it was used in training
d = {'Age': [65 / max_value], 'GenderCode': [1], 'PClass_2nd': [0], 'PClass_3rd': [0]}
y_result = model.predict(pd.DataFrame(data=d))
print("if him/her servive?", y_result)



