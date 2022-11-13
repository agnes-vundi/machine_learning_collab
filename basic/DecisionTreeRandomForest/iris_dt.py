import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz

df = pd.read_csv('iris.csv')

# make X and y
X = df.iloc[:, 0:4]
y = df.loc[:, ['Species']]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# train DecisionTreeClassifier
model = tree.DecisionTreeClassifier(max_depth = 3)
model.fit(X,y)

# visualize the tree
dot_data = export_graphviz(
            model,
            out_file =  None,
            feature_names = list(X.columns),
            class_names = df['Class'].unique(),
            filled = True,
            rounded = True)

graph = graphviz.Source(dot_data)
graph.render(filename = 'iris', format = 'png')

# Predicting the Test set results
y_pred = model.predict(X_test)

# make confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

# Calculate accuracy score
acc = accuracy_score(y_test, y_pred)
print (f'DC acc score: {acc:.2f} ')

# make three test flowers
new_flowers = [{'sepal length (cm)':5, 'sepal width (cm)':3.4, 'petal length (cm)':1.5, 'petal width (cm)':0.3},
               {'sepal length (cm)':6.5, 'sepal width (cm)':3, 'petal length (cm)':5.5, 'petal width (cm)':2.3},
               {'sepal length (cm)':5, 'sepal width (cm)':2.4, 'petal length (cm)':4, 'petal width (cm)':1.74}]
new_data = pd.DataFrame(new_flowers)

# predict with new data and create dataframe 
new_y = pd.DataFrame(model.predict(new_data))

# apply species information based on the prediction
new_y[1] = new_y[0].apply(lambda x: 'setosa' if x == 0  else ('versicolor' if x == 1 else 'virginica'))



