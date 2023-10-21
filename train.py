from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Загрузка датасета
wine = load_wine()

# Информация о датасете
print("Информация о датасете:")
print("Количество образцов: ", wine.data.shape[0])
print("Количество признаков: ", wine.data.shape[1])
print("Количество классов: ", len(wine.target_names))
print("Список имен классов: ", wine.target_names)

# Вывод первых 5 образцов
print("\nПервые 3 образца:")
for i in range(3):
    print("Образец ", i + 1, ":")
    print("Признаки: ", wine.data[i])
    print("Класс: ", wine.target_names[wine.target[i]])
    print()

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

# Создание DataFrame для тестовых данных
test_data = pd.DataFrame(x_test, columns=wine.feature_names)
test_data['target'] = y_test
# Сохранение данных в файлы CSV
test_data.to_csv('test_data.csv')

# Создание и обучение модели RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
# Сохранение модели
joblib.dump(model, 'wine_model.model')
print("Модель сохранена в файл 'wine_model.model'")

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