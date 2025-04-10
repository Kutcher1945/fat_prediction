import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px
import os

# --- Интерфейс приложения ---
st.set_page_config(page_title="Fat Prediction AI", layout="wide")

# 🚀 Отключаем GPU для TensorFlow на всякий случай
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- Кэширование моделей ---
@st.cache_resource
def load_models():
    rf_model = joblib.load("fat_model.pkl")
    keras_model = tf.saved_model.load("fat_model_tf")
    return rf_model, keras_model

rf_model, keras_model = load_models()

# --- Кэширование данных ---
@st.cache_data
def load_example():
    df = pd.read_csv('synthetic_medical_dataset.csv')
    df_encoded = pd.get_dummies(df, columns=['Пол', 'Препарат', 'Дозировка'])
    return df, df_encoded

df_example, df_encoded_example = load_example()

# --- Заголовок и описание проекта ---
st.title("💉 Fat Prediction AI — Прогноз жирности рук и ног на основе медикаментов")
st.markdown("""
Добро пожаловать!  
Это приложение позволяет на основе медицинских показателей пациента прогнозировать распределение жира по частям тела — **руки и ноги**.

**Как работает:**
- Вы указываете параметры пациента (возраст, пол, ИМТ, выбранный препарат и другие).
- Выбираете одну из обученных моделей (RandomForest или нейросеть).
- Получаете предсказание, а также интерактивные визуализации результата.

Приложение предназначено для:
- Врачей и специалистов по питанию.
- Исследователей в области медицины и здоровья.
- Пациентов, интересующихся влиянием факторов на распределение жира в теле.
""")

# --- Описание данных ---
with st.expander("ℹ️ Описание данных", expanded=True):
    st.markdown("""
    **Цель:**  
    На основе медицинских параметров пациента предсказать уровни жирности рук и ног.

    **Поля данных:**
    - Пол (Мужской / Женский)
    - Возраст
    - Индекс массы тела (ИМТ)
    - Препарат
    - Дозировка препарата
    - Физическая активность (часов в неделю)
    - Гормональный уровень
    - Метаболизм
    - Целевые переменные: Жирность рук, Жирность ног

    **Пример данных:**
    """)
    st.dataframe(df_example.head(), use_container_width=True)

# --- Сайдбар ---
st.sidebar.header("🔧 Настройки модели")
model_choice = st.sidebar.selectbox(
    "Выберите модель для предсказания:",
    ["RandomForest (pkl)", "Neural Network (TensorFlow SavedModel)"]
)

# --- Ввод данных пользователя ---
st.header("📋 Введите данные пациента:")

col1, col2 = st.columns(2)

with col1:
    Пол = st.selectbox("Пол", ["Мужской", "Женский"])
    Возраст = st.slider("Возраст", 18, 90, 30)
    ИМТ = st.slider("Индекс массы тела (ИМТ)", 15.0, 40.0, 22.0)
    Препарат = st.selectbox("Препарат", df_example["Препарат"].unique())

with col2:
    Дозировка = st.selectbox("Дозировка", df_example["Дозировка"].unique())
    Физическая_активность = st.slider("Физическая активность (часов в неделю)", 0, 20, 3)
    Гормональный_уровень = st.slider("Гормональный уровень", 0.0, 10.0, 5.0)
    Метаболизм = st.slider("Метаболизм", 0.5, 2.0, 1.0)

# --- Подготовка данных ---
input_dict = {
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

# Добавляем недостающие колонки с нулями
for col in df_encoded_example.drop(columns=['Жирность рук', 'Жирность ног']).columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Сохраняем порядок колонок
input_data = input_data[df_encoded_example.drop(columns=['Жирность рук', 'Жирность ног']).columns]
input_data_np = input_data.astype(np.float32).values

# --- Предсказание ---
if st.button("Сделать предсказание 🚀"):
    with st.spinner("Модель делает прогноз... 🔍"):
        if model_choice == "RandomForest (pkl)":
            prediction = rf_model.predict(input_data_np)[0]
        elif model_choice == "Neural Network (TensorFlow SavedModel)":
            prediction = keras_model.signatures["serve"](tf.constant(input_data_np))["output_0"].numpy()[0]
        else:
            prediction = [0, 0]

    # --- Вывод результатов ---
    st.subheader("🎯 Результаты предсказания:")
    st.markdown(f"""
    - **Жирность рук:** `{prediction[0]:.2f}`
    - **Жирность ног:** `{prediction[1]:.2f}`
    """)
    st.success("✅ Предсказание выполнено успешно!")

    # --- Подготовка данных для визуализации ---
    parts = ['Жирность рук', 'Жирность ног']
    values = [prediction[0], prediction[1]]
    df_plot = pd.DataFrame({'Часть тела': parts, 'Уровень жирности': values})

    # --- Bar Chart ---
    st.subheader("📊 Визуализация прогноза")
    st.markdown("Диаграмма показывает уровни жирности по каждой из частей тела.")
    fig_bar = px.bar(
        df_plot,
        x='Часть тела',
        y='Уровень жирности',
        color='Часть тела',
        text='Уровень жирности',
        color_discrete_sequence=['#4BA3C3', '#7BD389']
    )
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar.update_layout(
        yaxis_title='Уровень жирности',
        showlegend=False,
        margin=dict(t=30, b=10),
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Pie Chart ---
    st.subheader("🧩 Распределение жирности")
    st.markdown("Круговая диаграмма визуализирует вклад каждой части тела в общий прогноз.")
    fig_pie = px.pie(
        df_plot,
        names='Часть тела',
        values='Уровень жирности',
        color='Часть тела',
        color_discrete_sequence=['#4BA3C3', '#7BD389'],
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(margin=dict(t=30, b=10), height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Дополнительные визуализации ---
with st.expander("📈 Дополнительные визуализации исходных данных", expanded=False):
    st.markdown("### 🔥 Корреляционная матрица признаков")
    corr = df_encoded_example.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### 🖐️ Распределение жирности рук")
    fig_dist_arms = px.histogram(df_example, x='Жирность рук', nbins=20, color_discrete_sequence=['#4BA3C3'])
    st.plotly_chart(fig_dist_arms, use_container_width=True)

    st.markdown("### 🦵 Распределение жирности ног")
    fig_dist_legs = px.histogram(df_example, x='Жирность ног', nbins=20, color_discrete_sequence=['#7BD389'])
    st.plotly_chart(fig_dist_legs, use_container_width=True)

# --- Футер ---
st.markdown("---")
st.markdown("© 2025 Fat Prediction AI · Ваша система прогноза распределения жира 🧬")
