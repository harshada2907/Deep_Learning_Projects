# import libraries required
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

# load the dataset
df = pd.read_csv("Admission_Predict.csv")
print(df)

# check for null data
print(df.isnull().sum())

# check for duplicate values
print(df.duplicated().sum())

# info of the dataset
print(df.info())

df.drop(columns = ['Serial No.'], inplace = True)
print(df)

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# scaling the values
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()

model.add(Dense(7, activation = 'relu', input_dim = 7))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(1, activation = 'linear')) # whenever we are working with regression problems the output layer activation function will be linear

print(model.summary())

model.compile(loss = 'mean_squared_error', optimizer = 'Adam')

history = model.fit(X_train_scaled, y_train, epochs = 200, validation_split = 0.2)

y_pred = model.predict(X_test_scaled)
print(r2_score(y_test, y_pred))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()


