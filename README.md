Aquí tienes un ejemplo de cómo podrías estructurar el archivo README.md en Markdown para tu repositorio de GitHub que contiene la clase `ClusteringModel`. Este archivo proporcionará una descripción de la clase, cómo instalar las dependencias necesarias, y cómo usar la clase con un ejemplo.

```markdown
# ClusteringModel

The `ClusteringModel` class provides an interface for clustering analysis, including scaling, dimensionality reduction, and fitting various clustering models. It also supports evaluation of clustering results using silhouette scores and Davies-Bouldin index, and visualization using t-SNE.

## Installation

Ensure you have the following Python packages installed:
- `matplotlib`
- `seaborn`
- `umap-learn`
- `pandas`
- `numpy`
- `scikit-fuzzy`
- `scikit-learn`

You can install the required packages using pip:

```bash
pip install matplotlib seaborn umap-learn pandas numpy scikit-fuzzy scikit-learn
```

## Usage

Here's a step-by-step guide to using the `ClusteringModel` class:

1. **Import the necessary libraries and class:**

    ```python
    from sklearn.datasets import make_blobs
    from sklearn.manifold import TSNE
    from your_module import ClusteringModel  # Replace 'your_module' with the actual module name
    ```

2. **Generate a sample dataset:**

    ```python
    SEED = 42
    X, _ = make_blobs(n_samples=1000, centers=5, cluster_std=0.4)
    ```

3. **Create an instance of `ClusteringModel` with the dataset:**

    ```python
    clustering_model = ClusteringModel(X)
    ```

4. **Select the best model based on the silhouette score:**

    ```python
    best_model_info, df_models = clustering_model.select_best_model()
    ```

5. **Estimate the labels using the best model:**

    ```python
    labels = clustering_model.estimate_best_model()
    ```

6. **Print the best model and its parameters:**

    ```python
    print("Best selected model:")
    print(best_model_info)
    ```

7. **Initialize t-SNE for dimensionality reduction:**

    ```python
    tsne_model = clustering_model.fit_tsne()
    ```

8. **Plot the t-SNE results:**

    ```python
    clustering_model.plot_tsne()
    ```

## Methods

### `__init__(self, raw_tensor, verbose=False)`
Initializes the `ClusteringModel` with raw data and optional verbosity.

**Parameters:**
- `raw_tensor`: The raw data tensor for clustering.
- `verbose`: If True, prints additional information during initialization.

### `reduce_dimension(self, method='PCA', n_components=2, verbose=False)`
Reduces the dimensionality of the raw tensor using the specified method.

**Parameters:**
- `method`: Dimensionality reduction method to use ('PCA', 'UMAP', 'SVD').
- `n_components`: Number of components for reduction.
- `verbose`: If True, prints additional information during the reduction.

**Returns:**
- A dictionary with the reduced tensors for each scaler.

### `fit_models(self, use_reduction=True, use_scalers=True, verbose=False)`
Fits various clustering models to the data using optional scaling and dimensionality reduction.

**Parameters:**
- `use_reduction`: If True, apply dimensionality reduction before fitting models.
- `use_scalers`: If True, apply scaling before fitting models.
- `verbose`: If True, prints additional information during model fitting.

**Returns:**
- A list of dictionaries containing evaluation metrics for each model configuration.

### `select_best_model(self, metric='silhouette', use_reduction=True, use_scalers=True, n_min=5, n_max=7, verbose=False)`
Selects the best clustering model based on evaluation metrics and specified criteria.

**Parameters:**
- `metric`: Metric to use for selecting the best model ('silhouette' or 'davies_bouldin').
- `use_reduction`: If True, consider dimensionality reduction.
- `use_scalers`: If True, consider scaling.
- `n_min`: Minimum number of clusters to consider.
- `n_max`: Maximum number of clusters to consider.
- `verbose`: If True, prints additional information during model selection.

**Returns:**
- A dictionary with the best model, parameters, scaler, and reducer.
- A DataFrame with the filtered results.

### `estimate_best_model(self, use_reduction=True, use_scalers=True, verbose=False)`
Estimates the cluster labels using the best selected model.

**Parameters:**
- `use_reduction`: If True, apply dimensionality reduction using the best reducer.
- `use_scalers`: If True, apply scaling using the best scaler.
- `verbose`: If True, prints additional information during label estimation.

**Returns:**
- An array of cluster labels for each data point.

### `fit_tsne(self, n_components=2, perplexity=30, learning_rate=200)`
Fits t-SNE on the data using the best model and obtains 2D embeddings.

**Parameters:**
- `n_components`: Number of dimensions for t-SNE.
- `perplexity`: Perplexity parameter for t-SNE.
- `learning_rate`: Learning rate parameter for t-SNE.

**Raises:**
- ValueError if the best model has not been selected yet.

### `plot_tsne(self)`
Plots the t-SNE results to visualize the clustering results in 2D.

**Raises:**
- ValueError if t-SNE results or labels are not available. Please run `fit_tsne` first.

```

Este archivo `README.md` proporciona una guía clara para instalar las dependencias, usar la clase `ClusteringModel`, y explica cada uno de los métodos en detalle. Asegúrate de reemplazar `'your_module'` con el nombre real del archivo Python que contiene tu clase.