import os
os.environ["OMP_NUM_THREADS"] = '1'
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Read data from file to dataframe
df = pd.read_csv('iris.csv')

# Separate training data to be entered to the algorithm (X)
# Leave only species out, because we use it for result evaluation
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] #.copy(deep=False) # Could be used to avoid some warnings

# Scale the data for practise using Skleanr.Standardscaler
scaler = StandardScaler()

# Scale the data using the scaler
# Please note, the dataframe converts into a numpy array (not a dataframe anymore)
X = scaler.fit_transform(X)

# Empty list for SSE (Sum of Squared Errors) values
sse = []

# Try numbers from 1-10 as numbers of clusters 
for i in range(1, 11):
    # Parameters: number of clusters, "avoid random init problems"
    kmeans = KMeans(n_clusters=i, init='k-means++')
    # Train the model
    kmeans.fit(X)
    # Get and save the square distances sum (inertia_) for the model
    sse.append(kmeans.inertia_)
    
    # Plot a graph for SSE values to find the optimal number of clusters
plt.plot(range(1,11), sse)
plt.title('The Elbow graph')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Equared Errors')
plt.show()

# Parameters: number of clusters, "avoid random init problems" by using 'k-means++', to get same results with the teacher
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)

# Train model with cluster numbers and save the custers column
y_kmeans_predict = kmeans.fit_predict(X)

# Collect the input data, the prdicted clusters, and the known species into same dataframe
# Add original data to the dataframe
test_results = df

# Add Predicted species to the dataframe
test_results['predicted_species'] = y_kmeans_predict

# Convert 'Sales Ratign' column to numeric, i.e 'Good'-> 1 and 'Bad' to 0
test_results['predicted_species'] = test_results['predicted_species'].map({0: 'versicolor', 1: 'setosa', 2: 'virginica'})

# Print crosstab to see how the predicted clusters match with the known clusters (species)
cross_tab = pd.crosstab(test_results['predicted_species'], test_results['species'])
print('Crosstab:\n', cross_tab)

# Print also original species counts and predicted counts for comparison
print('\nSpecies:\n', test_results['species'].value_counts())
print('\nPredicted Species:\n', test_results['predicted_species'].value_counts())

# Inverse scale the scaled values back to original values
X = scaler.inverse_transform(X)

# Visualise (plot) the clusters just to see the clusters
plt.scatter(X[y_kmeans_predict == 0,0], X[y_kmeans_predict == 0,1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans_predict == 1,0], X[y_kmeans_predict == 1,1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans_predict == 2,0], X[y_kmeans_predict == 2,1], s=100, c='green', label='Cluster 3')

# Finalise the graph with information
plt.title('Clusters of flowers')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend()
plt.show()
