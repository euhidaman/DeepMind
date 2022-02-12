from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# model = Sequential([
#     Dense(5, input_shape=(3,), activation='relu'),
# ])

# In the above code, we were directly passing the layers in the `Sequential Constructor`
# The above code is similar to below :

model = Sequential()
model.add(Dense(5, input_shape=(3,)))
model.add(activation='relu')
