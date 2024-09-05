# ClusteringModel

<p align="center">
  <img src="https://www.freecodecamp.org/news/content/images/size/w2000/2020/09/analysis.png" alt="Clustering" width="300"/>
</p>

## Introduction

The objective of the `ClusteringModel` class is to provide a robust tool for performing clustering analysis on datasets. This class simplifies the application of various clustering algorithms, the selection of the best model based on evaluation metrics, and the visualization of results in two dimensions using t-SNE.

### Objective

In data analysis, clustering is a fundamental technique for grouping similar data and discovering hidden patterns. The `ClusteringModel` class is designed to simplify the clustering process by:

1. **Reducing Dimensionality**: It uses methods like PCA and UMAP to transform the data into a lower-dimensional space, facilitating the application of clustering models.
2. **Applying Clustering Models**: It implements several clustering algorithms, including K-means, DBSCAN, and others, to identify patterns in the data.
3. **Evaluating Models**: It uses evaluation metrics like the silhouette score and the Davies-Bouldin index to select the best clustering model.
4. **Visualizing Results**: It employs t-SNE to reduce dimensionality to two dimensions and visualize the clustering results.

### General Workflow

1. **Initialization**: The class is initialized with a dataset, optionally with configuration parameters.
2. **Dimensionality Reduction**: Depending on the configuration, dimensionality reduction techniques can be applied to enhance clustering effectiveness.
3. **Model Fitting**: Various clustering models are fitted to the data and evaluated using predefined metrics.
4. **Best Model Selection**: The clustering model that best fits the data is selected based on evaluation metrics.
5. **Label Estimation**: Clustering labels are estimated using the selected best model.
6. **Visualization**: t-SNE is used to reduce the data to two dimensions and visualize the clustering results.

This modular and flexible approach allows the clustering process to be adapted to different types of data and analytical goals, providing a powerful tool for exploratory analysis and the understanding of complex data.

```python
# Import necessary libraries
from sklearn.datasets import make_blobs
import pandas as pd

# Set a random seed for reproducibility
SEED = 42

# Generate a sample dataset with blobs
X, _ = make_blobs(n_samples=1000, centers=5, cluster_std=0.4)

# Exclusion Dictionary (Optional)
exclude_dict = {
    'scalers': ['StandardScaler'],
    'reducers': ['UMAP'],
    'models': ['KMeans']
}

# Initialize the clustering model
clustering_model = ClusteringModel(X, exclude_dict=exclude_dict)

# Select the best model based on the silhouette score
best_model_params, df_models_sorted = clustering_model.select_best_model()

# Print the model results
print(df_models_sorted)

# Print the best model parameters
print(best_model_params)

# Manually select a model (Optional)
manual_model_params = clustering_model.manual_selection_model(df_models_sorted, 3)

# Print the manually selected model parameters
print(manual_model_params)

# Estimate labels using the best model
labels = clustering_model.estimate_best_model(best_params=best_model_params)

# Initialize t-SNE for dimensionality reduction
clustering_model.fit_tsne()

# Plot the t-SNE results
clustering_model.plot_tsne(labels)
```
