import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#Prepare data 
df1 = pd.read_csv('gdp_figures.csv')
df1 = df1.dropna()
df2 = pd.read_csv('who_suicide_statistics.csv')
df2 = df2.dropna()

df = pd.merge(df2, df1)
df = df.drop_duplicates()

X = df[['country','sex','age','gdp_per_capita ($)']] # or df.loc[:, ['PClass','Age','GenderCode']]
y_orig = df[['suicides_no','population']]
population = df[['population']]

#scale suicide number to rate
y = y_orig['suicides_no']/y_orig['population']

# Split data into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test, population_train, population_test = train_test_split(X, y, population, test_size=0.2)

#dummy X values
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)


# train RandomForestClassifier
model = ensemble.RandomForestRegressor(max_depth=20, random_state=0)
model.fit(X_train,y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

#calculate and analyze regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print ('\nTest data metrics(Random Forest with max depth is 20):')
print("Mean Absolute Error is:" ,mae)
print("Mean Squared Error is:" ,mse)
print("Root Mean Squared Error is: ",rmse)
print("R2 score of the model is:" ,r2)