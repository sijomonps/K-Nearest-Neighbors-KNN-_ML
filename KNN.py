# =================================
# KNN Regression - House Price Prediction
# Dataset: kc_house_data.csv
# =================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Load Dataset
df = pd.read_csv("kc_house_data.csv")

print(df.head())


# Dataset Information
print(df.info())
print(df.describe())


# Drop unnecessary columns
df = df.drop(['id','date'], axis=1)


# Features and Target
X = df.drop('price', axis=1)
y = df['price']


# Feature Scaling (important for KNN)
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# Train KNN Model
knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(X_train, y_train)


# Prediction
y_pred = knn.predict(X_test)


# Model Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# Visualization
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("KNN: Actual vs Predicted Prices")

plt.show()


# Example Prediction
sample_house = X_scaled[0:1]

predicted_price = knn.predict(sample_house)

print("Predicted Price:", predicted_price)