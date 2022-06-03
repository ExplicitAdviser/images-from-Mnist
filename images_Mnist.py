import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

# загрузка обучающей и тестовой выборок
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

# преобразование выходных значений в векторы по категориям  - one hot encoding
# для цифры 5: y_train_cat: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

# формирование модели нейросети и вывод ее консоль
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary())      # вывод структуры нейросети в консоль

# компиляция нейросети с оптимизацией по Adam и критерием-категориальная кросс-энтропия
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# запуск процесса обучения: 80% - обучающая выборка, 20% - тестовая выборка
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

# подадим на вход тестовую выборку и получим
# критерий качества 'loss' и метрику 'accuracy'
model.evaluate(x_test, y_test_cat)

# проверка распознавания цифр
# подаем на вход тестовое n-изображение с индексом 22 в виде трехмерного тензора
# добавляем ось к двумерному изображению
n = 22
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print( res ) # 10 выходов в консоль
print(f"Распознанная цифра: {np.argmax(res)}") # с помощью функции argmax по индексу
# максимального значения среди 10 выходов выводим цифру

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Распознавание всей тестовой выборки
pred = model.predict(x_test) # пропустим через нейросеть всю тестовую выборку
pred = np.argmax(pred, axis=1)

print(pred.shape) # получим 10000 выходов и каждый выход является числом

print(pred[:20]) # первые 20 цифр которые предсказала нейросеть
print(y_test[:20]) # первые 20 цифр на самом деле

# Выделение неверных результатов
mask = pred == y_test # создадим маску, которая сравнивает и выводит булевы значения
print(mask[:10])

x_false = x_test[~mask] # выделяем из тестового множества элементы у которых mask является False
p_false = pred[~mask]

print(x_false.shape) # выводим количество неверно распознанных изображений из 10000

# Вывод первых 25 неверных результатов
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary)

plt.show()

# вывод первых 5 неверных результатов
for i in range(5):
    print("Значения сети: " +str(p_false[i]))
    plt.imshow(x_false[i], cmap = plt.cm.binary)
    plt.show()




