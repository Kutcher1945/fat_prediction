import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os

# 🚀 Отключаем GPU для TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Загружаем данные
df = pd.read_csv('synthetic_medical_dataset.csv')

# Подготовка данных
df_encoded = pd.get_dummies(df, columns=['Пол', 'Препарат', 'Дозировка'])
X = df_encoded.drop(columns=['Жирность рук', 'Жирность ног'])
y = df_encoded[['Жирность рук', 'Жирность ног']]

# Загружаем модели
rf_model = joblib.load('fat_model.pkl')
nn_model = load_model('fat_prediction_nn.h5', compile=False)

# Предсказания
rf_preds = rf_model.predict(X)
nn_preds = nn_model.predict(X)

# Функция оценки
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"\n📊 {name} Model Evaluation:")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    return mae, mse

# Оценка моделей
evaluate_model("RandomForest", y, rf_preds)
evaluate_model("Neural Network", y, nn_preds)

# Функция сохранения графиков
def plot_predictions(y_true, y_pred, title, filename):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].scatter(y_true['Жирность рук'], y_pred[:, 0], alpha=0.3)
    axs[0].plot([y_true['Жирность рук'].min(), y_true['Жирность рук'].max()],
                [y_true['Жирность рук'].min(), y_true['Жирность рук'].max()],
                color='red', linestyle='--')
    axs[0].set_xlabel('Фактическая жирность рук')
    axs[0].set_ylabel('Предсказанная жирность рук')
    axs[0].set_title(f'{title}: Руки')

    axs[1].scatter(y_true['Жирность ног'], y_pred[:, 1], alpha=0.3)
    axs[1].plot([y_true['Жирность ног'].min(), y_true['Жирность ног'].max()],
                [y_true['Жирность ног'].min(), y_true['Жирность ног'].max()],
                color='red', linestyle='--')
    axs[1].set_xlabel('Фактическая жирность ног')
    axs[1].set_ylabel('Предсказанная жирность ног')
    axs[1].set_title(f'{title}: Ноги')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ График сохранён: {filename}")

# Сохраняем графики
plot_predictions(y, rf_preds, "RandomForest", "randomforest_prediction.png")
plot_predictions(y, nn_preds, "Neural Network", "neural_network_prediction.png")
