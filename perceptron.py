import keras
from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.optimizers import Adam
import time

batchSize = 128
epochs = 50
neurons = 512

num_train = 60000
num_test = 10000

height, width, depth = 28, 28, 1
num_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(num_train, height * width)
X_test = X_test.reshape(num_test, height * width)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)

start_time = time.time()

model = keras.Sequential()

model.add(Dense(neurons, input_shape=(784, ), activation='relu'))
model.add(Dense(neurons, activation='relu'))
model.add(Dense(10, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=epochs, batch_size=batchSize)
print("Выводим результат обучения...")

start_ev_time = time.time()
model.evaluate(X_test, Y_test)
end_ev_time = time.time()

accuracy = model.evaluate(X_test,Y_test)

end_time = time.time()
interval = end_time - start_time
interval_ev = end_ev_time - start_ev_time
print(f'Точность {accuracy[1]*100:.2f}%')
print(f"Время Обучения: {interval:.6f} секунд")
print(f"Время Теста: {interval_ev:.6f} секунд")

model.save("perceptron.keras")