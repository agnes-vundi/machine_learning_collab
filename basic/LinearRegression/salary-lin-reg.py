import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv('salary.csv')

X = df.iloc[:,[0]] # experience
y = df.iloc[:,[1]] # salary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# train the model
model = LinearRegression()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# visualize
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs experience (testdata)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print ('\nTestdata metrics:')
print (f'mae: {mae}')
print (f'mse: {mse}')
print (f'rmse: {rmse}')
print(f'R2: {r2}') 

new_emp = pd.DataFrame([[7]], columns=['YearsExperience'])
print (f'New employee salary with 7 years of experience: {np.round(model.predict(new_emp),2)}\n' )
