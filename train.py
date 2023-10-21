from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_wine

# Загрузка датасета
data = load_wine()

# Информация о датасете
print("Информация о датасете:")
print("Количество образцов: ", data.data.shape[0])
print("Количество признаков: ", data.data.shape[1])
print("Количество классов: ", len(data.target_names))
print("Список имен классов: ", data.target_names)

# Вывод первых 5 образцов
print("\nПервые 5 образцов:")
for i in range(5):
    print("Образец ", i+1, ":")
    print("Признаки: ", data.data[i])
    print("Класс: ", data.target_names[data.target[i]])
    print()
