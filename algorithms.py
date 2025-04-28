import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv('C:\\Users\\fayc3\\OneDrive\\Desktop\\Semestre 2\\Machine Learning\\3 Machine learning algorithm\\DATASETS\\all_stocks_5yr.csv')

# Step 2: Preprocess the data
# Drop rows with missing values
data = data.dropna()
data = data.drop('Name',axis=1)

# Select a subset of rows for training (adjust as necessary)
data = data.iloc[:9000, :]  # First 3000 rows

# Convert 'date' column to datetime and extract numeric components
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek

# Drop the original 'date' column after extracting features
data = data.drop('date', axis=1)

# Separate features and target variable
X = data.drop('open', axis=1)  # Features (including 'volume' and new date features)
y = data['open']  # Target variable

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Separate features and target variable
X = data.drop('open', axis=1)  # Features
y = data['open']  # Target variable

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(name, model, X_train, X_test, y_train, y_test, mse_results, r2_results):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model: {name}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R^2 Score: {r2}\n")
    mse_results[name] = mse
    r2_results[name] = r2

# Initialize results dictionaries to store MSE and R² scores
mse_results = {}
r2_results = {}

# Step 4: Train and evaluate Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
evaluate_model('Decision Tree', dt_model, X_train, X_test, y_train, y_test, mse_results, r2_results)

# Step 5: Train and evaluate SVM
svm_model = SVR(kernel='linear', C=1.0)
evaluate_model('SVM', svm_model, X_train, X_test, y_train, y_test, mse_results, r2_results)

# Step 6: Train and evaluate Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
evaluate_model('Random Forest', rf_model, X_train, X_test, y_train, y_test, mse_results, r2_results)

# Step 7: Train and evaluate XGBoost Regressor
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
evaluate_model('XGBoost', xgb_model, X_train, X_test, y_train, y_test, mse_results, r2_results)

# Step 8: Plot the MSE scores for comparison
plt.figure(figsize=(10, 6))
plt.bar(mse_results.keys(), mse_results.values(), color=['blue', 'green', 'orange', 'red'])
plt.title('Model Comparison: Mean Squared Error (MSE)')
plt.ylabel('Mean Squared Error (MSE)')
plt.xlabel('Model')
plt.ylim(0, max(mse_results.values()) * 1.1)  # Set the y-axis limit slightly above the max MSE
plt.show()