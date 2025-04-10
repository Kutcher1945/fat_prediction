import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import os

# --- Интерфейс приложения ---
st.set_page_config(page_title="Fat & Eczema Prediction AI", layout="wide")

# 🚀 Отключаем GPU для TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- Кэширование моделей ---
@st.cache_resource
def load_models():
    rf_model = joblib.load("models/fat_prediction_rf.pkl")
    eczema_model = joblib.load("models/eczema_prediction_rf.pkl")
    keras_model = tf.keras.models.load_model("models/fat_prediction_nn.h5", compile=False)
    return rf_model, eczema_model, keras_model

rf_model, eczema_model, keras_model = load_models()

# --- Кэширование данных ---
@st.cache_data
def load_example():
    df = pd.read_csv('synthetic_medical_dataset_new.csv')
    df_encoded = pd.get_dummies(df, columns=['Пол', 'Препарат', 'Дозировка'])
    return df, df_encoded

df_example, df_encoded_example = load_example()

# --- Заголовок ---
st.title("🩺 Fat & Eczema Prediction AI — Прогноз жира и экземы на основе медицины")
st.markdown("""
Приложение прогнозирует:
- **Жирность рук и ног**
- **Вероятность наличия экземы**

Настройте параметры пациента и получите прогноз с визуализациями. 🔍
""")

# --- Описание данных ---
with st.expander("ℹ️ Описание данных", expanded=True):
    st.markdown("""
    **Поля данных:**
    - Пол (Мужской / Женский)
    - Возраст
    - Индекс массы тела (ИМТ)
    - Препарат
    - Дозировка препарата
    - Физическая активность (часов в неделю)
    - Гормональный уровень
    - Метаболизм
    - День (номер дня наблюдения)
    - Целевые переменные: Жирность рук, Жирность ног, Экзема
    """)
    st.dataframe(df_example.head(), use_container_width=True)

# --- Сайдбар ---
st.sidebar.header("🔧 Настройки модели")
model_choice = st.sidebar.selectbox(
    "Выберите модель для жирности:",
    ["RandomForest (pkl)", "Neural Network (Keras H5)"]
)

# --- Ввод данных пользователя ---
st.header("📋 Введите данные пациента:")

col1, col2 = st.columns(2)

with col1:
    Пол = st.selectbox("Пол", ["Мужской", "Женский"])
    Возраст = st.slider("Возраст", 18, 90, 30)
    ИМТ = st.slider("Индекс массы тела (ИМТ)", 15.0, 40.0, 22.0)
    Препарат = st.selectbox("Препарат", df_example["Препарат"].unique())
    День = st.slider("День наблюдения", int(df_example['День'].min()), int(df_example['День'].max()), int(df_example['День'].mean()))

with col2:
    Дозировка = st.selectbox("Дозировка", df_example["Дозировка"].unique())
    Физическая_активность = st.slider("Физическая активность (часов в неделю)", 0, 20, 3)
    Гормональный_уровень = st.slider("Гормональный уровень", 0.0, 10.0, 5.0)
    Метаболизм = st.slider("Метаболизм", 0.5, 2.0, 1.0)

# --- Подготовка данных ---
input_dict = {
    "День": День,
    "Возраст": Возраст,
    "ИМТ": ИМТ,
    "Физическая активность": Физическая_активность,
    "Гормональный уровень": Гормональный_уровень,
    "Метаболизм": Метаболизм,
    "Пол_" + Пол: 1,
    "Препарат_" + Препарат: 1,
    "Дозировка_" + Дозировка: 1
}

input_data = pd.DataFrame([input_dict])

for col in df_encoded_example.drop(columns=['Жирность рук', 'Жирность ног', 'Экзема']).columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[df_encoded_example.drop(columns=['Жирность рук', 'Жирность ног', 'Экзема']).columns]
input_data_np = input_data.astype(np.float32).values

# --- Предсказание ---
if st.button("Сделать предсказание 🚀"):
    with st.spinner("Модель делает прогноз... 🔍"):
        # Жирность
        if model_choice == "RandomForest (pkl)":
            prediction_fat = rf_model.predict(input_data_np)[0]
        elif model_choice == "Neural Network (Keras H5)":
            prediction_fat = keras_model.predict(input_data_np)[0]
        else:
            prediction_fat = [0, 0]

        # Экзема
        prediction_eczema_prob = eczema_model.predict_proba(input_data_np)[0][1]
        prediction_eczema_class = eczema_model.predict(input_data_np)[0]

    # --- Результаты ---
    st.subheader("🎯 Результаты предсказания:")
    st.markdown(f"""
    - **Жирность рук:** `{prediction_fat[0]:.2f}`
    - **Жирность ног:** `{prediction_fat[1]:.2f}`
    - **Экзема (вероятность):** `{prediction_eczema_prob * 100:.1f}%`
    - **Экзема (класс):** `{'Да' if prediction_eczema_class == 1 else 'Нет'}`
    """)
    st.success("✅ Предсказание выполнено успешно!")

    # --- Визуализация жирности ---
    st.subheader("📊 Визуализация прогноза жирности")
    df_fat_plot = pd.DataFrame({'Часть тела': ['Жирность рук', 'Жирность ног'], 'Уровень жирности': prediction_fat})
    fig_bar = px.bar(df_fat_plot, x='Часть тела', y='Уровень жирности', color='Часть тела', text='Уровень жирности',
                     color_discrete_sequence=['#4BA3C3', '#7BD389'])
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar.update_layout(yaxis_title='Уровень жирности', showlegend=False, margin=dict(t=30, b=10), height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Визуализация экземы ---
    st.subheader("🧩 Вероятность экземы")
    df_eczema_plot = pd.DataFrame({'Состояние': ['Экзема', 'Здоров'], 'Вероятность': [prediction_eczema_prob, 1 - prediction_eczema_prob]})
    fig_pie = px.pie(df_eczema_plot, names='Состояние', values='Вероятность', color='Состояние',
                     color_discrete_sequence=['#FF6B6B', '#6BCB77'], hole=0.4)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(margin=dict(t=30, b=10), height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Дополнительные визуализации ---
with st.expander("📈 Дополнительные визуализации исходных данных", expanded=True):
    st.markdown("### 🔥 Тепловая карта корреляции")
    corr = df_encoded_example.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### 📊 Динамика экземы по дням")
    eczema_over_time = df_example.groupby('День')['Экзема'].mean().reset_index()
    fig_line = px.line(eczema_over_time, x='День', y='Экзема', markers=True, line_shape='spline',
                       labels={'Экзема': 'Средняя вероятность экземы'})
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("### 🧠 Интерпретация модели (SHAP values)")
    with st.spinner("Готовим интерпретацию модели..."):
        explainer = shap.TreeExplainer(eczema_model)
        features = df_encoded_example.drop(columns=['Жирность рук', 'Жирность ног', 'Экзема'])
        shap_values = explainer.shap_values(features)
    
        # Создаём интерактивный bar-плот через matplotlib
        shap.plots.bar(shap.Explanation(values=shap_values[1], base_values=explainer.expected_value[1], data=features, feature_names=features.columns.tolist()), show=False)
    
        # Захватываем matplotlib figure для Streamlit
        st.pyplot(plt.gcf())
        plt.clf()


# --- Футер ---
st.markdown("---")
st.markdown("© 2025 Fat & Eczema Prediction AI · Аналитика жира и кожных заболеваний 🧬")
