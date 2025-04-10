import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# ✅ Force CPU (to avoid CuDNN errors)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load dataset
df = pd.read_csv('synthetic_medical_dataset.csv')

# Encode categorical features
df_encoded = pd.get_dummies(df, columns=['Пол', 'Препарат', 'Дозировка'])

# Define features and targets
X = df_encoded.drop(columns=['Жирность рук', 'Жирность ног'])
y = df_encoded[['Жирность рук', 'Жирность ног']]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- RandomForest Model -----
rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
rf.fit(X_train, y_train)
joblib.dump(rf, 'fat_model.pkl')
print("✅ RandomForest model saved as fat_model.pkl")

# ----- Neural Network Model -----
# Build neural network
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2)  # Output layer: 2 targets
])

model.compile(optimizer='adam', loss='mse')

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
model.fit(X_train, y_train, validation_split=0.2, epochs=50, callbacks=[early_stop], verbose=1)

# Save the model
model.save('fat_prediction_nn.h5')
print("✅ Neural network model saved as fat_prediction_nn.h5")
