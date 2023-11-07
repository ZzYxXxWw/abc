import pandas as pd import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_samples, silhouette_score import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage from scipy.spatial.distance import pdist
# Load the Iris dataset iris = load_iris()
X = iris.data y = iris.target
 
# (a) Identifying the Number of Clusters # Calculate the linkage matrix
linkage_matrix = linkage(X, method='ward') # You can choose different linkage methods # Use a dendrogram to identify the number of clusters
dendrogram(linkage_matrix) plt.title('Dendrogram') plt.xlabel('Samples') plt.ylabel('Distance') plt.show()
# Choose the number of clusters based on dendrogram n_clusters = 3 # Example: 3 clusters
# Perform hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=n_clusters) cluster_memberships = hierarchical.fit_predict(X)
# Select two features for the scatter plot (change as needed) feature1 = 0
feature2 = 1

plt.scatter(X[cluster_memberships == 0, feature1], X[cluster_memberships == 0, feature2], c='red', label='Cluster 1')
plt.scatter(X[cluster_memberships == 1, feature1], X[cluster_memberships == 1, feature2], c='blue', label='Cluster 2')
plt.scatter(X[cluster_memberships == 2, feature1], X[cluster_memberships == 2, feature2], c='green', label='Cluster 3')
plt.xlabel(iris.feature_names[feature1]) plt.ylabel(iris.feature_names[feature2]) plt.legend()
 
plt.title('Scatter Plot of Clustered Objects') plt.show()
# (d) Creating a Barplot for Silhouette Coefficients silhouette_avg = silhouette_score(X, cluster_memberships)
sample_silhouette_values = silhouette_samples(X, cluster_memberships) sns.barplot(x=sample_silhouette_values, y=range(len(sample_silhouette_values))) plt.xlabel('Silhouette Coefficient')
plt.ylabel('Cluster')

plt.title(f'Silhouette Coefficients for {n_clusters} Clusters (Avg: {silhouette_avg:.2f})') plt.show()
