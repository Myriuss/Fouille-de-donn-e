import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the training and test datasets
train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

# Assuming you've already done preprocessing and feature engineering
# For demonstration, let's drop non-numeric columns
non_numeric_cols = train_data.select_dtypes(exclude=['int', 'float']).columns
X_train = train_data.drop(['SalePrice'] + list(non_numeric_cols), axis=1)
y_train = train_data['SalePrice']

# One-hot encode categorical variables
X_train = pd.get_dummies(X_train)

# Train your model (Random Forest Regressor is just an example, use any model you prefer)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test data
# Preprocess test data similarly before making predictions
# Assuming preprocessing steps are the same for both train and test data
X_test = test_data.drop(non_numeric_cols, axis=1)
X_test = pd.get_dummies(X_test)
predictions = model.predict(X_test)

# Create a DataFrame for submission
submission_df = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)
