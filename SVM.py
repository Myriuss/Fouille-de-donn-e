import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger les données à partir du fichier CSV
data = pd.read_csv("student_admission_dataset/student_admission_dataset.csv")

# Diviser les données en fonctionnalités et étiquettes
X = data[['GPA', 'SAT_Score']]
y = data['Admission_Status']

# Remplacer les labels par des valeurs numériques
y = y.replace({'Accepted': 1, 'Rejected': 0, 'Waitlisted': 0}, inplace=False)

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les fonctionnalités
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialiser le classifieur SVM
svm_classifier = SVC(kernel='linear')

# Entraîner le modèle SVM
svm_classifier.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = svm_classifier.predict(X_test)

# Créer un graphique
plt.figure(figsize=(10, 6))

# Plot les étudiants acceptés en bleu
plt.scatter(X_test[y_pred == 1][:, 0], X_test[y_pred == 1][:, 1], color='blue', label='Accepted')
# Plot les étudiants non acceptés en rouge
plt.scatter(X_test[y_pred == 0][:, 0], X_test[y_pred == 0][:, 1], color='red', label='Not Accepted')

# Ajouter les étiquettes et le titre
plt.xlabel('GPA')
plt.ylabel('SAT Score')
plt.title('Répartition des étudiants acceptés et non acceptés')
plt.legend()
plt.grid(True)
plt.show()
