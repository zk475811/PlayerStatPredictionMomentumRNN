from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import numpy as np

# take data for last 4 games to detect hot or cold streaks
timesteps = 4
input_dim = 3

# features are player, home or away, age
# need to add more features to list to fill out predictive capacity
x_train = np.array([[[1, 0, 25],
		     [1, 0, 25],
		     [1, 1, 25],
		     [1, 0, 25]]])

# fantasy points
y_train = np.array([[29, 33, 33, 42]])

# need to add test data, this just to show network learns
x_test = x_train
y_test = y_train

# expects shape of (batch_size, timesteps, input_dim)
model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, input_dim)))
# outputs a prediction for every game the last prediction is for the next game that hasn't happened yet
model.add(Dense(timesteps, activation='relu'))

# mean square error regression
model.compile(optimizer='rmsprop', loss='mse')

model.fit(x_train, y_train, batch_size=1, epochs=4000)
# batch size will be expanded once more data is collected
score = model.evaluate(x_test, y_test, batch_size=1)

# the most recent three games plus data for upcoming matchup
last_four_games = np.array([[[1, 0, 25],
			     [1, 1, 25],
			     [1, 0, 25],
			     [1, 0, 25]]])
print()
# predict stats for next game
print(model.predict(last_four_games, batch_size=1))
