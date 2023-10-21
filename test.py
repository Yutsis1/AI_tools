from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Загрузка данных из файла CSV в DataFrame
test_data = pd.read_csv('test_data.csv')
x_test = test_data.drop('target', axis=1).values
y_test = test_data['target'].values

# Загрузка модели из файла
model = joblib.load('wine_model.model')

# Выполнение предсказаний на тестовых данных
y_pred = model.predict(x_test)

if __name__ == "__main__":
    # Прогнозирование классов для тестовых данных
    y_pred = model.predict(x_test)

    # Оценка точности модели
    accuracy = accuracy_score(y_test, y_pred)
    print("Точность модели: {:.2f}%".format(accuracy * 100))

    # Отображение матрицы пересечений
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Матрица пересечений')
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.show()
