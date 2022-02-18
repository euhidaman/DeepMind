from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras_visualizer import visualizer

model = Sequential([
    Dense(16, activation='relu', input_shape=(1, 5)),
    # `glorot_uniform` is for Xavier distribution using uniform distribution. This is the default.
    Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
    Dense(2, activation='softmax')
])


model.summary()
