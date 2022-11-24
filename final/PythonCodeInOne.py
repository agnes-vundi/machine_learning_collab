import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble


#Prepare data 
df1 = pd.read_csv('gdp_figures.csv')
df1 = df1.dropna()
df2 = pd.read_csv('who_suicide_statistics.csv')
df2 = df2.dropna()

df = pd.merge(df2, df1)
df = df.drop_duplicates()

X = df[['sex','age','gdp_per_capita ($)']] 
y_orig = df[['suicides_no','population']]
population = df[['population']]

#scale suicide number to rate
y = y_orig['suicides_no']/y_orig['population']

# Split data into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test, population_train, population_test = train_test_split(X, y, population, test_size=0.2)

# Save the original format data in readable format for the later testing phase
X_test_orig = X_test


#dummy X values
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Only scale at LinearRegression
XL_train = X_train, XL_test = X_test                       
# after make new data frame to scale, there is a error 'too many values to unpack', it was fine with separate code.
max_gdp = X['gdp_per_capita ($)'].max()
XL_train['gdp_per_capita ($)'] = XL_train['gdp_per_capita ($)'] / max_gdp
XL_test['gdp_per_capita ($)'] = XL_test['gdp_per_capita ($)'] / max_gdp

#training the model
model = linear_model.LinearRegression()
model.fit(XL_train, y_train)

#testing model
yL_pred = model.predict(XL_test)

# Create a new dataframe for the results
test_results = pd.DataFrame()
test_results = X_test_orig
test_results['Real output'] = y_test
test_results['Predicted output'] = yL_pred
test_results['population'] = population_test
test_results['Real suicide'] = test_results['Real output'] * test_results['population']
test_results['Predicted suicide'] = test_results['Predicted output'] * test_results['population'] 


#calculate and analyze regression metrics
mae = mean_absolute_error(y_test, yL_pred)
mse = mean_squared_error(y_test, yL_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, yL_pred)
print ('\nLinear Regression:')
print ('\nTest data metrics(without country as an independent variable ):')
print("Mean Absolute Error is:" ,mae)
print("Mean Squared Error is:" ,mse)
print("Root Mean Squared Error is: ",rmse)
print("R2 score of the model is:" ,r2)


# train DecisionTreeClassifier
model_DT = DecisionTreeRegressor(max_depth=30,
                              random_state=0)
model_DT.fit(X_train,y_train)

# visualize the tree
dot_data = export_graphviz(model_DT, out_file='decisiontree.dot') 

# Predicting the Test set results
yDT_pred = model_DT.predict(X_test)

#calculate and analyze regression metrics
mae = mean_absolute_error(y_test, yDT_pred)
mse = mean_squared_error(y_test, yDT_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, yDT_pred)

print ('\nDecision Tree:')
print ('Test data metrics(max depth is 30):')
print("Mean Absolute Error is:" ,mae)
print("Mean Squared Error is:" ,mse)
print("Root Mean Squared Error is: ",rmse)
print("R2 score of the model is:" ,r2)



# train RandomForestClassifier
model_RF = ensemble.RandomForestRegressor(max_depth=20, random_state=0)
model_RF.fit(X_train,y_train)

# Predicting the Test set results
yRF_pred = model_RF.predict(X_test)

#calculate and analyze regression metrics
mae = mean_absolute_error(y_test, yRF_pred)
mse = mean_squared_error(y_test, yRF_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, yRF_pred)

print ('\nRandom Forest:')
print ('Test data metrics(Random Forest with max depth is 20):')
print("Mean Absolute Error is:" ,mae)
print("Mean Squared Error is:" ,mse)
print("Root Mean Squared Error is: ",rmse)
print("R2 score of the model is:" ,r2)

























