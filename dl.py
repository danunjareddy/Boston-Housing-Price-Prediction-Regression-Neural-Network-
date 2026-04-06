import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(train_data, train_targets, epochs=100, batch_size=16, verbose=0)

mse, mae = model.evaluate(test_data, test_targets, verbose=0)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
baseline_mae = model.evaluate(test_data, test_targets, verbose=0)[1]
importances = []

for i in range(test_data.shape[1]):
    temp_test = test_data.copy()
    np.random.shuffle(temp_test[:, i])
    shuffled_mae = model.evaluate(temp_test, test_targets, verbose=0)[1]
    importances.append(shuffled_mae - baseline_mae)

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nFeature Importance (Impact on MAE):")
print(importance_df.sort_values(by='Importance', ascending=False))