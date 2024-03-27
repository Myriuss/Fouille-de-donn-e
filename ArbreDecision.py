from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Charger les données du cancer du sein
breast_cancer = load_breast_cancer()

# Créer un DataFrame pandas pour faciliter la manipulation des données
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['target'] = breast_cancer.target

# Sélectionner les caractéristiques radius_mean et concave points_mean
X = data[['mean radius', 'mean concave points']]
y = data['target']

# Entraîner un modèle CART
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X, y)

# Définir les limites du graphique avec un espace supplémentaire
x_min, x_max = 0, 30
y_min, y_max = 0, 0.2

# Générer les coordonnées de la grille
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Prédire les classes pour chaque point de la grille
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Créer un graphique en 2D
plt.figure(figsize=(10, 6))

# Tracer la région de décision
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')

# Tracer les points bénins en rouge
plt.scatter(X[y == 1]['mean radius'], X[y == 1]['mean concave points'], c='red', label='Benign', marker='o', edgecolors='k')

# Tracer les points malins en bleu
plt.scatter(X[y == 0]['mean radius'], X[y == 0]['mean concave points'], c='blue', label='Malignant', marker='o', edgecolors='k')

plt.title('Cancer du sein : Radius Mean vs Concave Points Mean')
plt.xlabel('Radius Mean')
plt.ylabel('Concave Points Mean')
plt.grid(True)
plt.legend()
plt.show()
