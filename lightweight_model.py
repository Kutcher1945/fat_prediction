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

# ✅ Force CPU для стабильности
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Create models directory
os.makedirs('models', exist_ok=True)

# Load updated dataset
df = pd.read_csv('synthetic_medical_dataset_new.csv')

# Encode categorical features
df_encoded = pd.get_dummies(df, columns=['Пол', 'Препарат', 'Дозировка'])

# Features and Targets
X = df_encoded.drop(columns=['Жирность рук', 'Жирность ног', 'Экзема'])
y_regression = df_encoded[['Жирность рук', 'Жирность ног']]
y_classification = df_encoded['Экзема']

# Split into train and test sets
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Early stopping для обеих нейросетей
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ============================
# RandomForest Regression Model
# ============================
print("🚀 Training RandomForest regression model...")
rf_reg = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=30,
    max_depth=7,
    min_samples_leaf=5,
    random_state=42
))
rf_reg.fit(X_train, y_train_reg)
joblib.dump(rf_reg, 'models/fat_prediction_rf.pkl')
print("✅ RandomForest regression model saved as models/fat_prediction_rf.pkl")

# ============================
# RandomForest Classification Model (Eczema)
# ============================
print("🚀 Training RandomForest classification model (eczema)...")
rf_clf = RandomForestClassifier(
    n_estimators=30,
    max_depth=7,
    min_samples_leaf=5,
    random_state=42
)
rf_clf.fit(X_train, y_train_clf)
joblib.dump(rf_clf, 'models/eczema_prediction_rf.pkl')
print("✅ RandomForest classification model saved as models/eczema_prediction_rf.pkl")

# ============================
# Neural Network Regression Model
# ============================
print("🚀 Training Neural Network regression model...")
nn_reg = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2)  # Output: 2 targets
])
nn_reg.compile(optimizer='adam', loss='mse')

nn_reg.fit(X_train, y_train_reg, validation_split=0.2, epochs=20, batch_size=16, callbacks=[early_stop], verbose=1)
nn_reg.save('models/fat_prediction_nn.h5')
print("✅ Neural network regression model saved as models/fat_prediction_nn.h5")

# ===== Convert Neural Network Regression to TFLite =====
print("🚀 Converting Neural Network regression to TFLite...")
converter_reg = tf.lite.TFLiteConverter.from_keras_model(nn_reg)
converter_reg.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_reg = converter_reg.convert()
with open('models/fat_prediction_nn.tflite', 'wb') as f:
    f.write(tflite_model_reg)
print("✅ Neural network regression quantized model saved as models/fat_prediction_nn.tflite")

# ============================
# Neural Network Classification Model (Eczema)
# ============================
print("🚀 Training Neural Network classification model (eczema)...")
nn_clf = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output: probability of eczema
])
nn_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

nn_clf.fit(X_train, y_train_clf, validation_split=0.2, epochs=20, batch_size=16, callbacks=[early_stop], verbose=1)
nn_clf.save('models/eczema_prediction_nn.h5')
print("✅ Neural network classification model saved as models/eczema_prediction_nn.h5")

# ===== Convert Neural Network Classification to TFLite =====
print("🚀 Converting Neural Network classification to TFLite...")
converter_clf = tf.lite.TFLiteConverter.from_keras_model(nn_clf)
converter_clf.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_clf = converter_clf.convert()
with open('models/eczema_prediction_nn.tflite', 'wb') as f:
    f.write(tflite_model_clf)
print("✅ Neural network classification quantized model saved as models/eczema_prediction_nn.tflite")

print("🎉 All models are trained and saved successfully in 'models/' folder!")
