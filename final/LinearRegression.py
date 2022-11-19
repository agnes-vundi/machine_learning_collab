import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



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


#dummy and scale X values
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
max_gdp = X['gdp_per_capita ($)'].max()
X_train['gdp_per_capita ($)'] = X_train['gdp_per_capita ($)'] / max_gdp
X_test['gdp_per_capita ($)'] = X_test['gdp_per_capita ($)'] / max_gdp

#training the model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

#testing model
y_pred = model.predict(X_test)

# Create a new dataframe for the results
test_results = pd.DataFrame()
test_results = X_test_orig
test_results['Real output'] = y_test
test_results['Predicted output'] = y_pred
test_results['population'] = population_test
test_results['Real suicide'] = test_results['Real output'] * test_results['population']
test_results['Predicted suicide'] = test_results['Predicted output'] * test_results['population']


#calculate and analyze regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print ('\nTest data metrics(without country as an independent variable ):')
print("Mean Absolute Error is:" ,mae)
print("Mean Squared Error is:" ,mse)
print("Root Mean Squared Error is: ",rmse)
print("R2 score of the model is:" ,r2)


























