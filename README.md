
# ClusteringModel

<p align="center">
  <img src="https://www.freecodecamp.org/news/content/images/size/w2000/2020/09/analysis.png" alt="Clustering" width="300"/>
</p>

## Introducción

El objetivo de la clase `ClusteringModel` es proporcionar una herramienta robusta para realizar análisis de clustering en conjuntos de datos. Esta clase facilita la aplicación de varios algoritmos de clustering, la selección del mejor modelo basado en métricas de evaluación y la visualización de resultados en dos dimensiones mediante t-SNE.

### Objetivo

En el análisis de datos, el clustering es una técnica fundamental para agrupar datos similares y descubrir patrones ocultos. La clase `ClusteringModel` está diseñada para simplificar el proceso de clustering al:

1. **Reducir la Dimensionalidad**: Utiliza métodos como PCA y UMAP para transformar los datos a un espacio de menor dimensión, facilitando la aplicación de modelos de clustering.
2. **Aplicar Modelos de Clustering**: Implementa varios algoritmos de clustering, incluyendo K-means, DBSCAN, y otros, para identificar patrones en los datos.
3. **Evaluar Modelos**: Utiliza métricas de evaluación como el índice de silueta y el índice de Davies-Bouldin para seleccionar el mejor modelo de clustering.
4. **Visualizar Resultados**: Emplea t-SNE para reducir la dimensionalidad a dos dimensiones y visualizar los resultados del clustering.

### Funcionamiento General

1. **Inicialización**: La clase se inicializa con un conjunto de datos, y opcionalmente con parámetros de configuración.
2. **Reducción de Dimensionalidad**: Dependiendo de la configuración, se pueden aplicar técnicas de reducción de dimensionalidad para mejorar la efectividad del clustering.
3. **Ajuste de Modelos**: Se ajustan diversos modelos de clustering a los datos, y se evalúan utilizando métricas predeterminadas.
4. **Selección del Mejor Modelo**: Se selecciona el modelo de clustering que mejor se ajusta a los datos basándose en las métricas de evaluación.
5. **Estimación de Etiquetas**: Se estiman las etiquetas de clustering utilizando el mejor modelo seleccionado.
6. **Visualización**: Se utiliza t-SNE para reducir los datos a dos dimensiones y visualizar los resultados del clustering.

Este enfoque modular y flexible permite adaptar el proceso de clustering a diferentes tipos de datos y objetivos analíticos, proporcionando una herramienta poderosa para el análisis exploratorio y la comprensión de datos complejos.


```python
# Importar librerías necesarias
from sklearn.datasets import make_blobs
import pandas as pd

# Establecer una semilla aleatoria para reproducibilidad
SEED = 42

# Generar un conjunto de datos de ejemplo con blobs
X, _ = make_blobs(n_samples=1000, centers=5, cluster_std=0.4)

# Diccionario de Exclusión (Opcional)
exclude_dict = {
    'scalers': ['StandardScaler'],
    'reducers': ['UMAP'],
    'models': ['KMeans']
}

# Inicializar el modelo de clustering
clustering_model = ClusteringModel(X, exclude_dict=exclude_dict)

# Seleccionar el mejor modelo basado en el puntaje de silueta
best_model_params, df_models_sorted = clustering_model.select_best_model()

# Imprimir los resultados de los modelos
print(df_models_sorted)

# Imprimir los parámetros del mejor modelo
print(best_model_params)

# Seleccionar manualmente un modelo (Opcional)
manual_model_params = clustering_model.manual_selection_model(df_models_sorted, 3)

# Imprimir los parámetros del modelo manualmente seleccionado
print(manual_model_params)

# Estimar las etiquetas usando el mejor modelo
labels = clustering_model.estimate_best_model(best_params=best_model_params)

# Inicializar t-SNE para reducción de dimensionalidad
clustering_model.fit_tsne()

# Graficar los resultados de t-SNE
clustering_model.plot_tsne(labels)
```
