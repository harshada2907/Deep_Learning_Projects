# import the libraries

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import accuragicy_score
import warnings 
warnings.filterwarnings("ignore")

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(X_train[0])

print(X_test.shape)

print(y_train)

plt.imshow(X_train[0])
plt.show()

X_train = X_train / 255
X_test = X_test / 255

print(X_train[0])

# model building

model = Sequential()

model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax')) # when we want more than 2 output classes as output then we use the softmax activation function

print(model.summary())

# model compiling

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

# model fitting

history = model.fit(X_train, y_train, epochs = 10, validation_split = 0.2)

y_prob = model.predict(X_test)

print(y_prob)

y_pred = y_prob.argmax(axis = 1)

print(accuracy_score(y_test, y_pred))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

plt.imshow(X_test[0])
plt.show()

print(model.predict(X_test[0].reshape(1, 28, 28)).argmax(axis = 1))