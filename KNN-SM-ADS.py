import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Social_Network_Ads/Social_Network_Ads.csv")

# Diviser les données en fonction de l'achat
purchased_data = data[data['Purchased'] == 1]
not_purchased_data = data[data['Purchased'] == 0]

# Créer le graphe
plt.figure(figsize=(10, 6))

# Afficher les points pour les achats et les non-achats
plt.scatter(purchased_data['Age'], purchased_data['EstimatedSalary'], color='green', label='Purchased', marker='o')
plt.scatter(not_purchased_data['Age'], not_purchased_data['EstimatedSalary'], color='red', label='Not Purchased', marker='x')

# Ajouter des titres et une légende
plt.title('Age vs Estimated Salary')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Afficher le graphe
plt.grid(True)
plt.show()
