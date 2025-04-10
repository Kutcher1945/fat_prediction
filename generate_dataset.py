import pandas as pd
import numpy as np
import random

# Устанавливаем сид для воспроизводимости
random.seed(42)
np.random.seed(42)

# Параметры генерации
n_samples = 5000

# Категориальные признаки
sexes = ['Мужской', 'Женский']
drugs = ['Препарат A', 'Препарат B', 'Препарат C']
dosages = ['Низкая', 'Средняя', 'Высокая']

# Генерация данных
data = {
    'Пол': np.random.choice(sexes, n_samples),
    'Возраст': np.random.randint(18, 90, n_samples),
    'ИМТ': np.round(np.random.uniform(15.0, 40.0, n_samples), 1),
    'Препарат': np.random.choice(drugs, n_samples),
    'Дозировка': np.random.choice(dosages, n_samples),
    'Физическая активность': np.random.randint(0, 20, n_samples),
    'Гормональный уровень': np.round(np.random.uniform(0.0, 10.0, n_samples), 1),
    'Метаболизм': np.round(np.random.uniform(0.5, 2.0, n_samples), 2),
}

df = pd.DataFrame(data)

# Генерация целевых переменных: жирность рук и ног
df['Жирность рук'] = (
    0.2 * df['ИМТ'] +
    0.1 * df['Гормональный уровень'] -
    0.15 * df['Физическая активность'] +
    np.where(df['Дозировка'] == 'Высокая', 2, 0) +
    np.random.normal(0, 1, n_samples)
)

df['Жирность ног'] = (
    0.25 * df['ИМТ'] +
    0.05 * df['Гормональный уровень'] -
    0.1 * df['Физическая активность'] +
    np.where(df['Дозировка'] == 'Средняя', 1.5, 0) +
    np.random.normal(0, 1, n_samples)
)

# Генерация целевой переменной: Экзема
def generate_eczema(row):
    risk = 0.2
    if row['Жирность рук'] > 6 or row['Жирность ног'] > 6:
        risk += 0.3
    risk += (row['Гормональный уровень'] / 20)
    if row['Метаболизм'] < 1.0:
        risk += 0.2
    risk += np.random.normal(0, 0.1)  # Шум
    return int(np.clip(risk, 0, 1) > 0.5)

df['Экзема'] = df.apply(generate_eczema, axis=1)

# Сохраняем датасет
df.to_csv('synthetic_medical_dataset.csv', index=False)

print("✅ Новый synthetic_medical_dataset.csv с полем 'Экзема' успешно создан!")
