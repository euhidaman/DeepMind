from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------
# Video Link : https://www.youtube.com/watch?v=UkzhouEk6uY

train_labels = []
train_samples = []

# Creating Dummy sample training data
# Data test case :
# An experimental drug was tested on totally 2100 individuals
# between ages 13 to 100yrs. Half were <65, and half were >65
# 95% of patients 65yrs or older, had side effects to a certain drug
# 95% of patients under 65yrs, had No side effects to a certain drug
for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)  # storing first 1000
    train_labels.append(0)

    random_older = randint(65, 100)
    train_samples.append(random_older)  # storing first 50
    train_labels.append(1)


for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)  # storing second 1000
    train_labels.append(1)

    random_older = randint(65, 100)
    train_samples.append(random_older)  # storing second 50
    train_labels.append(0)

# for i in train_samples:
#     print(i)

# for i in train_labels:
#     print(i)

# converting training data in to numpy array,
# bcz `Keras expects data in numpy array`
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_samples = scaler.fit_transform(
    (train_samples)
    # `reshape` : Converts 1D array to 2D, bcz fit_transform doesn't accept 1D array
    .reshape(-1, 1)
)

for i in scaled_training_samples:
    print(i)

# ---------------------------------------------------------
# Video Link: https: // youtu.be/_N5kpSMDf4o

model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(Adam(lr=.0001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(scaled_training_samples, train_labels,
          batch_size=10, epochs=20, shuffle=True, verbose=2)
