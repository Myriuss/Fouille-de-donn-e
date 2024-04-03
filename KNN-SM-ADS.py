import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Charger les données
data = pd.read_csv("Social_Network_Ads/Social_Network_Ads.csv")

# Diviser les données en fonction de l'achat
purchased_data = data[data['Purchased'] == 1]
not_purchased_data = data[data['Purchased'] == 0]

# Séparer les caractéristiques et la cible
X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialiser le classifieur k-NN
k = 5  # Nombre de voisins à considérer
knn = KNeighborsClassifier(n_neighbors=k)

# Entraîner le modèle k-NN
knn.fit(X_train_scaled, y_train)

# Prédire les valeurs sur l'ensemble de test
y_pred = knn.predict(X_test_scaled)

# Créer le graphe
plt.figure(figsize=(10, 6))

# Afficher les points pour les achats et les non-achats dans l'ensemble de test
plt.scatter(X_test[y_test == 1]['Age'], X_test[y_test == 1]['EstimatedSalary'], color='green', label='Purchased', marker='o')
plt.scatter(X_test[y_test == 0]['Age'], X_test[y_test == 0]['EstimatedSalary'], color='red', label='Not Purchased', marker='x')

# Ajouter des titres et une légende
plt.title('Age vs Estimated Salary (k-NN)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Afficher le graphe
plt.grid(True)
plt.show()
