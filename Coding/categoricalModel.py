#CREATING A MODEL USING NEURAL NETWORK (FRAMEWORK )

###### SOURCE WHERE I AM GETTING THE CODE HELP FROM 
#https://data-flair.training/blogs/python-chatbot-project/

import os
import pandas as pd

#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import numpy as np
import TrainingModel as tm
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.optimizers import gradient_descent_v2
from matplotlib import pyplot as plt #to plot the graph


##BUILDING A NEURAL NETWORK
#3 layers. First layer 3120 neurons, second layer 3120 neurons and 3rd output layer contains number of neurons to predict output with sigmoid, which is 585
#3120 inputs(from length of input) -> [3120 hidden nodes] -> [3120 2ndhidden nodes] -> 585 outputs
model = Sequential()
model.add(Dense(3120, input_shape=(len(tm.X_train[0]),), activation='relu'))
#dropout helps to prevent overfitting
model.add(Dropout(0.5))
model.add(Dense(3120, activation='relu'))
model.add(Dropout(0.5))
#softmax for categorical
model.add(Dense(len(tm.y_train[0]), activation='softmax'))
model.summary()

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = gradient_descent_v2.SGD(learning_rate = 0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
#fitting the model
#chatbotModel = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size = 8, validation_data=(tm.X_test, tm.y_test), verbose=1)
#to make it go process faster, it it turned into a numpy array
categoricalChatbotModel = model.fit(tm.X_train, tm.y_train, epochs=250, batch_size = 5, validation_data=(tm.X_test, tm.y_test), verbose=1)
#saving the model 
model.save('categoricalChatbot_model3.h5', categoricalChatbotModel)
print("model created")

#this is to make a graph from the accuracy and loss of the chatbot model made
plt.plot(categoricalChatbotModel .history['accuracy'], label='training set accuracy')
plt.plot(categoricalChatbotModel .history['loss'], label = 'training set loss')
plt.show()
plt.legend()
plot_model(model(), show_shapes=True)