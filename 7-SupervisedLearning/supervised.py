import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(2, activation='sigmoid')
])

model.compile(Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Supervised data sample presented as : weight in pounds, height in inches
train_sample = [[150, 67], [130, 60], [
    200, 65], [125, 52], [230, 72], [101, 70]]

# Supervised data labels presented as --> 0:male , 1:female
train_labels = [1, 1, 0, 1, 0, 0]

model.fit(train_sample, train_labels,
          batch_size=3, epochs=20, shuffle=True, verbose=2)
