from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import numpy as np

timesteps = 4
input_dim = 3

x_train = np.array([[[1, 0, 25],
									   [1, 0, 25],
									   [1, 1, 25],
									   [1, 0, 25]]])

y_train = np.array([[29, 33, 33, 42]])

x_test = x_train
y_test = y_train

# expects shape of (batch_size, timesteps, input_dim)
model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, input_dim)))
model.add(Dense(timesteps, activation='relu'))

# mean square error regression
model.compile(optimizer='rmsprop', loss='mse')

model.fit(x_train, y_train, batch_size=1, epochs=4000)
score = model.evaluate(x_test, y_test, batch_size=1)

last_four_games = np.array([[[1, 0, 25],
														 [1, 1, 25],
														 [1, 0, 25],
														 [1, 0, 25]]])
print()
print(model.predict(last_four_games, batch_size=1))
