import os
import joblib
import onnx
import tf2onnx
import tensorflow as tf
from tensorflow.keras.models import load_model
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import pandas as pd

# 🚀 Отключаем GPU для TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Путь к моделям
keras_model_path = "fat_prediction_nn.h5"
sklearn_model_path = "fat_model.pkl"

# Загружаем модели
print("ℹ️ Загружаем модели...")
keras_model = load_model(keras_model_path, compile=False)
sklearn_model = joblib.load(sklearn_model_path)

# Dummy input для экспорта
df = pd.read_csv('synthetic_medical_dataset.csv')
df_encoded = pd.get_dummies(df, columns=['Пол', 'Препарат', 'Дозировка'])
X = df_encoded.drop(columns=['Жирность рук', 'Жирность ног'])
sample_input = X[:1].astype(np.float32)

# 1️⃣ TensorFlow SavedModel
print("🚀 Конвертация Keras модели в TensorFlow SavedModel...")
keras_savedmodel_path = "fat_model_tf"
keras_model.export(keras_savedmodel_path)
print(f"✅ TensorFlow SavedModel сохранён в: {keras_savedmodel_path}/")

# 2️⃣ Keras → ONNX с tf.function (обход бага output_names)
print("🚀 Конвертация Keras модели в ONNX...")

# Определяем tf.function с входной сигнатурой
spec = (tf.TensorSpec(sample_input.shape, tf.float32, name="input"),)

@tf.function(input_signature=[spec[0]])
def model_func(x):
    return keras_model(x)

# Конвертация модели через tf.function
onnx_model, _ = tf2onnx.convert.from_function(model_func, input_signature=spec, opset=13)
onnx_model_path = "fat_model.onnx"

with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Keras модель сохранена в ONNX формате: {onnx_model_path}")

# 3️⃣ Scikit-Learn → ONNX
print("🚀 Конвертация RandomForest модели в ONNX...")
initial_type = [("float_input", FloatTensorType([None, sample_input.shape[1]]))]
sklearn_onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type)
sklearn_onnx_path = "fat_model_rf.onnx"

with open(sklearn_onnx_path, "wb") as f:
    f.write(sklearn_onnx_model.SerializeToString())

print(f"✅ RandomForest модель сохранена в ONNX формате: {sklearn_onnx_path}")

print("🎉 Все модели успешно конвертированы в продакшен-форматы!")
