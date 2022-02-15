# Valid Padding --> means, NO padding --> Data/Image dimensions reduce

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

model_valid = Sequential([  # The input data has dimensions of 20x20
    Dense(16, input_shape=(20, 20, 3), activation='relu'),
    Conv2D(32, kernel_size=(3, 3), activation='relu',
           padding='valid'),  # First, we're filering it with 3x3 filter
    Conv2D(64, kernel_size=(5, 5), activation='relu',
           padding='valid'),  # Then with 5x5 filter
    Conv2D(128, kernel_size=(7, 7), activation='relu',
           padding='valid'),  # Then with 7x7 filter
    Flatten(),
    # Finally, we're outputting it to the two out labels
    Dense(2, activation='softmax')
])


model_valid.summary()
