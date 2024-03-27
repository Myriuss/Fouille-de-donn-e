from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données du cancer du sein
breast_cancer = load_breast_cancer()

# Créer un DataFrame pandas pour faciliter la manipulation des données
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['target'] = breast_cancer.target

# Sélectionner les caractéristiques radius_mean et concave points_mean
X = data[['mean radius', 'mean concave points']]
y = data['target']

# Créer un graphique en 2D
plt.figure(figsize=(10, 6))

# Tracer les points bénins en bleu
plt.scatter(X[y == 1]['mean radius'], X[y == 1]['mean concave points'], c='blue', label='Benign', marker='o', edgecolors='k')

# Tracer les points malins en rouge
plt.scatter(X[y == 0]['mean radius'], X[y == 0]['mean concave points'], c='red', label='Malignant', marker='o', edgecolors='k')

plt.title('Cancer du sein : Radius Mean vs Concave Points Mean')
plt.xlabel('Radius Mean')
plt.ylabel('Concave Points Mean')
plt.grid(True)
plt.legend()
plt.show()
