# This file must be run
# directly from the terminal, by typing in the command manually

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras_visualizer import visualizer

model = Sequential([
    Dense(5, activation='relu', input_shape=(3,)),
    Dense(6, activation='relu'),
    Dense(2, activation='softmax')
])

visualizer(model, filename='ANN_graph', format='png', view=True)
