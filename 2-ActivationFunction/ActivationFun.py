from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# model = Sequential([
#     Dense(5, input_shape=(3,), activation='relu'),
# ])

# In the above code, we were directly passing the layers in the `Sequential Constructor`
# The above code is similar to below :

model = Sequential()
model.add(Dense(5, input_shape=(3,)))
model.add(Activation('relu'))

# or, directly put the activation function in one line

# model.add(layers.Dense(2, activation="relu", input_shape=(4,)))

model.summary()
