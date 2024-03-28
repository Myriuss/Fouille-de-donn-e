import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Charger l'ensemble de données Iris
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # Sélectionner petal length et petal width

# Choisissez un nombre arbitraire de clusters (par exemple, 3) pour commencer
k = 3

# Effectuer le clustering K-means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Afficher le nuage de points avec les clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=100, c='red', label='Centroids')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Clustering des Iris (petal length vs petal width)')
plt.legend()
plt.show()

# Calculer le score de silhouette pour évaluer la qualité du clustering
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f"Score de silhouette moyen : {silhouette_avg}")
