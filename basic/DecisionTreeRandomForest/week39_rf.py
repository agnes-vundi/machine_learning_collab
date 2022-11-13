import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
from sklearn.tree import export_graphviz
import graphviz

df = pd.read_csv('titanic.csv')

# make X and y
X = df.iloc[:, 0:3]
y = df.loc[:, ['Survived']]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# train RandomForestClassifier
model = ensemble.RandomForestClassifier(max_depth=5)
model.fit(X_train,y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# make confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

# Calculate accuracy score
acc = accuracy_score(y_test, y_pred)
print ("In this Random Forest Classifier: ")
print (f'RF acc score: {acc:.2f} ')

# make two test people, 1 for female, 0 for male.
two_people= [{'PClass':1,'Age':17,'Gender':1},
             {'PClass':3,'Age':17,'Gender':0}]
new_data = pd.DataFrame(two_people)

# predict with new data and create dataframe 
new_y = pd.DataFrame(model.predict(new_data))

print("According to prediction, Rose and Jack will...(1 for survive, 0 for dead)")
print(new_y.to_string(index=False,header=False))