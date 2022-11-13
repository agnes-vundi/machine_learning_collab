import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Read data to pandas dataframe
df = pd.read_csv('sales_data.csv')

# Create input data (X) and output data (y)
X = df[['Weekday','Seller']] 
y = df[['Sales Rating']]

# Split data into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save the original format data in readable format for the later testing phase
X_test_orig = X_test


# Dummy variables avoid the 'dummy variable trap'
X_train = pd.get_dummies(X_train, prefix=['Seller'], columns=['Seller'])
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, prefix=['Seller'], columns=['Seller'])
X_test = pd.get_dummies(X_test, drop_first=True)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train.values.ravel())

# Predicting outputs with X_test as inputs
y_pred = model.predict(X_test)

# Create empty dataframe for the results
test_results = pd.DataFrame()

# Get original test values from X_test (values are saved in X_test_orig)
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






