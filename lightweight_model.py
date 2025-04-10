import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ✅ Force CPU (to avoid GPU / CuDNN errors)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load dataset
df = pd.read_csv('synthetic_medical_dataset.csv')

# Encode categorical features
df_encoded = pd.get_dummies(df, columns=['Пол', 'Препарат', 'Дозировка'])

# Features and Targets
X = df_encoded.drop(columns=['Жирность рук', 'Жирность ног', 'Экзема'])
y_regression = df_encoded[['Жирность рук', 'Жирность ног']]
y_classification = df_encoded['Экзема']

# Split into train and test sets
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# ===== RandomForest Model for Regression =====
print("🚀 Training lightweight RandomForest regression model...")
rf_reg = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=30,
    max_depth=7,
    min_samples_leaf=5,
    random_state=42
))
rf_reg.fit(X_train, y_train_reg)
joblib.dump(rf_reg, 'fat_model.pkl')
print("✅ RandomForest regression model saved as fat_model.pkl")

# ===== RandomForest Model for Classification =====
print("🚀 Training lightweight RandomForest classifier model (for eczema)...")
rf_clf = RandomForestClassifier(
    n_estimators=30,
    max_depth=7,
    min_samples_leaf=5,
    random_state=42
)
rf_clf.fit(X_train, y_train_clf)
joblib.dump(rf_clf, 'eczema_model.pkl')
print("✅ RandomForest classification model saved as eczema_model.pkl")

# ===== Neural Network Model (Lightweight) =====
print("🚀 Training lightweight Neural Network model for regression...")
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2)  # Output: 2 values (fat arms, fat legs)
])
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train_reg, validation_split=0.2, epochs=20, batch_size=16, callbacks=[early_stop], verbose=1)

model.save('fat_prediction_nn.h5')
print("✅ Neural network regression model saved as fat_prediction_nn.h5")

# ===== Convert to TFLite (quantized) =====
print("🚀 Converting Neural Network to quantized TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('fat_prediction_nn.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Neural network quantized model saved as fat_prediction_nn.tflite")

print("🎉 All models are trained and saved successfully!")
