from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import f1_score
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Загрузка данных из файла CSV в DataFrame
test_data = pd.read_csv('test_data.csv')
x_test = test_data.drop('target', axis=1).values
y_test = test_data['target'].values

# Загрузка модели из файла
model = joblib.load('wine_model.model')

# Выполнение предсказаний на тестовых данных
y_pred = model.predict(x_test)
# Расчет метрики F1
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 score:", f1)

# Запись метрики F1 в файл
with open('pred.txt', 'w') as file:
    file.write(str(list(y_pred)))

if __name__ == "__main__":
    # Отображение матрицы пересечений
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Матрица пересечений')
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.show()
