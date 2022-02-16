# Same padding --> Zeroes are present around the edges --> Data/Image dimensions stay the same

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

model_valid = Sequential([
    Dense(16, input_shape=(20, 20, 3), activation='relu'),  # input of 20x20
    Conv2D(32, kernel_size=(3, 3), activation='relu',
           padding='same'),  # --> Convolutional layer of 3x3
    # Next --> MaxPool layer of 2x2, must be exactly after convolutional layer output
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(64, kernel_size=(5, 5), activation='relu',
           padding='same'),  # --> Convolutional layer of 5x5
    Flatten(),
    Dense(2, activation='softmax')  # Finally output to 2 nodes
])


model_valid.summary()
