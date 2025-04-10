import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 🚀 Отключаем GPU для TensorFlow

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Загружаем модели
rf_model = joblib.load('fat_model.pkl')
nn_model = load_model('fat_prediction_nn.h5', compile=False)

# Пример новых данных пациента
input_data = pd.DataFrame([{
    'ID пациента': 1234,
    'Возраст': 45,
    'ИМТ': 27.5,
    'День': 14,
    'Пол_Ж': 0,
    'Пол_М': 1,
    'Препарат_Анаболик': 0,
    'Препарат_Антидепрессант': 1,
    'Препарат_Глюкокортикостероид': 0,
    'Препарат_Гормональная терапия': 0,
    'Препарат_Контрольная группа': 0,
    'Дозировка_Высокая': 1,
    'Дозировка_Низкая': 0,
    'Дозировка_Средняя': 0
}])

# Предсказание RandomForest
rf_prediction = rf_model.predict(input_data)[0]

# Предсказание нейросети
nn_prediction = nn_model.predict(input_data)[0]

print(f"✅ RandomForest prediction: Жирность рук: {rf_prediction[0]:.2f}, Жирность ног: {rf_prediction[1]:.2f}")
print(f"✅ Neural Network prediction: Жирность рук: {nn_prediction[0]:.2f}, Жирность ног: {nn_prediction[1]:.2f}")
