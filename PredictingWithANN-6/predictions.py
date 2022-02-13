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
    train_labels.append(1)  # Appending class label

    random_older = randint(65, 100)
    train_samples.append(random_older)  # storing second 50
    train_labels.append(0)  # Appending class label

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

# ---------------------------------------------------------
# Video Link: https: // youtu.be/_N5kpSMDf4o

model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(Adam(learning_rate=.0001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# The below code has `validation_split=0.20`, which splits the given test data into 20% validation
model.fit(scaled_training_samples, train_labels, validation_split=0.20,
          batch_size=10, epochs=20, shuffle=True, verbose=2)

# But, we can alternatively provide `validation data` separately too. As shown below -->

# valid_set = [(sample,label),(sample,label), ... ,(sample,label)]
# model.fit(scaled_training_samples, train_labels, validation_split=valid_set,batch_size=10, epochs=20, shuffle=True, verbose=2)


# ----------------------------Creating dummy test data --------------------

test_samples = []

# dummy 100 Data for patients above age 65
for i in range(50):
    random_younger_test = randint(13, 64)
    test_samples.append(random_younger_test)

    random_older_test = randint(65, 100)
    test_samples.append(random_older_test)

test_samples = np.array(test_samples)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_test_samples = scaler.fit_transform(
    (test_samples)
    .reshape(-1, 1)
)

# ---------------------- Predicting with Test Data ------------------------
# Video Link : https://www.youtube.com/watch?v=Z0KVRdE_a7Q
predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)

for i in predictions:
    print(i)
