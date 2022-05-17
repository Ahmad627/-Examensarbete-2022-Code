from itertools import count

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
from tensorflow import keras
from keras import layers
import time
import wandb
import math
import numpy as np

#inställningar
nsamples = 1000     # Number of samples to use as a dataset
val_ratio = 0.2     # Percentage of samples that should be held for validation set
test_ratio = 0.2    # Percentage of samples that should be held for test set
tflite_model_name = 'sine_model'  # Will be given .tflite suffix
c_model_name = 'sine_model'       # Will be given .h suffix

# Generera några slumpmässiga prover
np.random.seed(1234)
x_values = np.random.uniform(low=0, high=(2 * math.pi), size=nsamples)
plt.plot(x_values)

# Skapa en brusig sinusvåg med dessa värden 
y_values = np.sin(x_values) + (0.1 * np.random.randn(x_values.shape[0]))
plt.plot(x_values, y_values, '.')

# Split the dataset into training, validation, and test sets
val_split = int(val_ratio * nsamples)
test_split = int(val_split + (test_ratio * nsamples))
x_val, x_test, x_train = np.split(x_values, [val_split, test_split])
y_val, y_test, y_train = np.split(y_values, [val_split, test_split])

# Kontrollera att våra uppdelningar summerar korrekt
assert(x_train.size + x_val.size + x_test.size) == nsamples

# Plot the data in each partition in different colors:
plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_test, y_test, 'r.', label="Test")
plt.plot(x_val, y_val, 'y.', label="Validate")
plt.legend()
plt.show()

t1_start = time.process_time()

# Create a model
model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(1,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

# Visa modell
model.summary()

# Add optimizer, loss function, and metrics to model and compile it
model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])

# Train model
history = model.fit(x_train, y_train, epochs=500, batch_size=100, validation_data = (x_val, y_val))

# Plot the training history
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Rita prediktiva mot faktiska värden
predictions = model.predict(x_test)

plt.clf()
plt.title("Comparison of predictions to actual values")
plt.plot(x_test, y_test, 'b.', label='Actual')
plt.plot(x_test, predictions, 'r.', label='Prediction')
plt.legend()
plt.show()

# Standardavvikelse
def variance(predictions):
    n = len(predictions)
    mean = sum(predictions) / n
    return sum((x - mean) ** 2 for x in predictions) / (n)

def stdev(predictions):
    var = variance(predictions)
    std_dev = math.sqrt(var)
    return std_dev

print("Standard avvikelse")
print(stdev(predictions))

## beräkna standardavvikelsen för felet
def RSE(x_test, predictions):

    # x-test: Actual value
    # predictions: predicated values

    x_test = np.array(x_test)
    predictions = np.array(predictions)
    RSS = np.sum(np.square(x_test - predictions))

    rse = math.sqrt(RSS / (len(x_test) - 2))
    return rse

print("standard deviation of error ")
print(RSE(x_test, predictions))
plt.errorbar(x_test, predictions,  linestyle='None', marker='^')
#plt.show()


print("                                      ")
t1_stop = time.process_time()
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
#print("sin(x): ")
#print(x_test)
#print (y_test)
#stop = time.time()
#print(f"Training time: {stop - start}s")

