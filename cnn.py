import keras
from keras.api.datasets import mnist
from keras.api.utils import to_categorical
import time

height, width, depth = 28, 28, 1
num_classes = 10

batch_size = 32
num_epochs = 50
hidden_size = 512

(X_train, y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], height, width, depth)
X_train = X_train.astype('float32')
X_train /= 255

X_test = X_test.reshape(X_test.shape[0], height, width, depth)
X_test = X_test.astype('float32')
X_test /= 255

Y_train = to_categorical(y_train, num_classes=num_classes)
Y_test = to_categorical(Y_test, num_classes=num_classes)

start_time = time.time()

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, depth)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, depth)))
# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, depth)))
# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size,epochs=num_epochs,verbose=1,validation_split=0.1)
model.save_weights(f'{len(model.layers)}-{num_epochs}-{hidden_size}.weights.h5')
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
