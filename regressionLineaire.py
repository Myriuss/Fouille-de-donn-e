import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("auto-mpg.csv")

X = data[['displacement']]
y = data['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de régression linéaire
model = LinearRegression()

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
predictions = model.predict(X_test)

# Calculer l'erreur de la régression (par exemple, l'erreur quadratique moyenne)
mse = np.mean((y_test - predictions) ** 2)
print("Mean Squared Error:", mse)

# Visualiser les résultats
#plt.scatter(y_test, predictions)

plt.scatter(X_test, y_test)

#plt.scatter(predictions, y_test)  # Inverser l'ordre des arguments

plt.xlabel("Predicted Mileage")  # Conserver le label de l'axe x
plt.ylabel("Mileage")  # Conserver le label de l'axe y

plt.plot(X_test, model.predict(X_test))

plt.title("Predicted Mileage vs Mileage")  # Conserver le titre du graphique
plt.show()
print(data.head())
