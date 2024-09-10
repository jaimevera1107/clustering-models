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
4. **Visualizing Results**: It employs methods to reduce dimensionality to two dimensions and visualize the clustering results.

### General Workflow

1. **Initialization**: The class is initialized with a dataset, optionally with configuration parameters.
2. **Dimensionality Reduction**: Depending on the configuration, dimensionality reduction techniques can be applied to enhance clustering effectiveness.
3. **Model Fitting**: Various clustering models are fitted to the data and evaluated using predefined metrics.
4. **Best Model Selection**: The clustering model that best fits the data is selected based on evaluation metrics.
5. **Label Estimation**: Clustering labels are estimated using the selected best model.
6. **Visualization**: t-SNE/PCA is used to reduce the data to two dimensions and visualize the clustering results.

This modular and flexible approach allows the clustering process to be adapted to different types of data and analytical goals, providing a powerful tool for exploratory analysis and the understanding of complex data.

```python
# ----------------------------------------------------------------------
# Models Estimation Example (default)
# ----------------------------------------------------------------------

# Set a random seed for reproducibility
SEED = 42

# Generate a sample dataset with blobs
X, _ = make_blobs(
    n_samples=5000,
    centers=5,
    cluster_std=0.4
)

# Initializing the class
clustering_model = ClusteringModel(X_input=X)

# Select the best model
best_model_params, df_models_sorted = clustering_model.select_best_model()

# Estimate the labels using the best model
labels = clustering_model.estimate_best_model(best_params=best_model_params)

# Visualize the clusters in two dimensions
clustering_model.visualize_reduced_data(labels=labels)
# ----------------------------------------------------------------------
```

```python
# ----------------------------------------------------------------------
# Models Estimation Example (with own Dicts)
# ----------------------------------------------------------------------

# Set a random seed for reproducibility
SEED = 42

# Generate a sample dataset with blobs
X, _ = make_blobs(
    n_samples=5000,
    centers=5,
    cluster_std=0.4
)

# Exclusion Dict (Optional)
exclusion_dict = {
    'scalers': ['StandardScaler'],
    'reducers': ['UMAP'],
    'models': ['KMeans']
}

# Parameters Dict (Optional)
params_dict = {
    'DBSCAN': {
        'eps': [round(i * 0.1, 1) for i in range(4, 6)],
        'min_samples': list(range(16, 32 * 2, 16))
    },
    'KMeans': {'n_clusters': list(range(2, 7))},
    'AgglomerativeClustering': {'n_clusters': list(range(2, 7))},
    'MeanShift': {'bandwidth': list(range(2, 7))},
    'FuzzyCMeans': {
        'n_clusters': list(range(2, 7)),
        'm': list(range(1, 4))
    }
}

# Initializing the class
clustering_model = ClusteringModel(
    X_input=X,
    exclude_dict=exclusion_dict,
    dict_params=params_dict,
)

# Select the best model
best_model_params, df_models_sorted = clustering_model.select_best_model()

# Estimate the labels using the best model
labels = clustering_model.estimate_best_model(best_params=best_model_params)

# Visualize the clusters in two dimensions
clustering_model.visualize_reduced_data(labels=labels)
# ----------------------------------------------------------------------
```

```python
# ----------------------------------------------------------------------
# Models Estimation Example (with Categorical Data Models, Default)
# ----------------------------------------------------------------------

data = {
    'feature1': np.random.rand(1000) * 100,
    'feature2': np.random.rand(1000) * 100,
    'category1': np.random.choice(['A', 'B', 'C', 'D'], size=1000),
    'category2': np.random.choice(['X', 'Y', 'Z'], size=1000)
}
df = pd.DataFrame(data)

def get_pd_categorical_indices(df):
        """
        Given a DataFrame, extracts the indices of categorical columns.
        
        :param df: pandas DataFrame
        :return: list of indices of categorical columns
        """
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Get the indices of categorical columns
        categorical_indices = [df.columns.get_loc(col) for col in categorical_cols]
        
        return categorical_indices

# Get the indices of categorical columns
cat_indices = get_pd_categorical_indices(df)

# Convert DataFrame to a NumPy array
X = df.to_numpy()

# Initializing the class
clust_model_cat = ClusteringModel(
    X_input=X,
    use_categorical=True,
    categorical_cols=cat_indices
)

# Select the best model
best_model_params, df_models_sorted = clust_model_cat.select_best_model()

# Estimate the labels using the best model
labels = clust_model_cat.estimate_best_model(best_params=best_model_params)
# ----------------------------------------------------------------------
```
