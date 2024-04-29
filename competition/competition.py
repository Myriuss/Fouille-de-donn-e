import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Charger les données
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Supprimer les colonnes inutiles ou non informatives
train_data.drop(columns=['id', 'CustomerId', 'Surname'], inplace=True)
test_data.drop(columns=['id', 'CustomerId', 'Surname'], inplace=True)

# Diviser les données d'entraînement en features et target
X = train_data.drop(columns=['Exited'])
y = train_data['Exited']

# Diviser les données d'entraînement en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing des features
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Entraîner le modèle de régression logistique
clf = LogisticRegression()

model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', clf)])

model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de validation
y_val_pred = model.predict(X_val)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_val, y_val_pred)
print("Accuracy on validation set:", accuracy)

# Prétraitement des données de test
X_test = test_data

# Faire des prédictions sur les données de test
predictions = model.predict_proba(X_test)[:, 1]

# Créer un DataFrame pour les prédictions avec les identifiants corrects
final_submission = pd.DataFrame({'id': sample_submission['id'], 'Exited': predictions})

# Enregistrer le fichier de soumission final au format CSV
final_submission.to_csv('final_submission.csv', index=False)
