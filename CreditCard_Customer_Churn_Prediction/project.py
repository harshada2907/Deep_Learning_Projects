# import the libraries required 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# printing the data to study
df = pd.read_csv("Churn_Modelling.csv")
print(df)

# checking for the null values in the data
print(df.isnull().sum())

# print the info about the data in order to understand the data
print(df.info())

# check for the duplicated rows in the data
print(df.duplicated().sum())

print(df['Exited'].value_counts())

print(df['Geography'].value_counts())

print(df['Gender'].value_counts())

df.drop(columns = ['RowNumber', 'CustomerId', 'Surname'], inplace = True)

print(df)

df = pd.get_dummies(df, columns = ['Geography', 'Gender'], drop_first = True)

print(df)


# splitting the data for training and testing
X = df.drop(columns = ['Exited'])
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(X)
print(y)

print(X_train.shape)
print(X_test.shape)

# scaling the features down

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)
print(X_test_scaled)

# making the sequential model
model = Sequential()

model.add(Dense(11, activation = 'relu', input_dim = 11))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

print(model.summary())

model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs = 100, validation_split = 0.2)

# print(history.history)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

print(model.layers[0].get_weights())
print(model.layers[1].get_weights())

y_log = model.predict(X_test_scaled)

y_pred = np.where(y_log > 0.5, 1, 0)

print(accuracy_score(y_test, y_pred))




