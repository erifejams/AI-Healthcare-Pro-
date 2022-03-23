import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDOneClassSVM, SGDRegressor

#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


import tensorflow as tf
from tensorflow import keras as ks
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Concatenate, GlobalAveragePooling1D, Dropout
from keras import Input, Model, layers
from keras.layers.attention import Attention
import TrainingModel as tm
from keras.models import load_model
from keras.optimizers import gradient_descent_v2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""
model = ks.Sequential()
model.add(layers.Embedding(input_length = 3120, input_dim=10000, output_dim=10000))    
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.LSTM(64))
model.add(layers.Dense(587, activation="softmax"))
model.compile(optimizer=ks.optimizers.SGD(learning_rate=(0.55)), loss=ks.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.summary()
"""


#3 layers. First layer 5000 neurons, second layer 2000 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with relu
model = ks.Sequential()
model.add(Dense(5000, input_shape=(len(tm.X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(tm.y_train[0]), activation='relu'))
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = gradient_descent_v2.SGD(learning_rate=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#fitting and saving the model 
chatbotModel = model.fit(np.array(tm.X_train), np.array(tm.y_train), epochs=200, batch_size = 5, validation_data=(tm.X_test, tm.y_test), verbose=1)
model.save('chatbot_model.h5', chatbotModel)
print("model created")