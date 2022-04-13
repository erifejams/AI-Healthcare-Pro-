#CREATING A MODEL USING NEURAL NETWORK (FRAMEWORK )

import os
import pandas as pd

#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import numpy as np
import TrainingModel as tm

from keras.models import Sequential
from keras.layers import Dense, Dropout
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
#sigmoid for binary
model.add(Dense(len(tm.y_train[0]), activation='sigmoid'))
model.summary()

# Compiling the model.
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

#fitting the model
#chatbotModel = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size = 8, validation_data=(tm.X_test, tm.y_test), verbose=1)
#to make it go process faster, it it turned into a numpy array
chatbotModel = model.fit(tm.X_train, tm.y_train, epochs=5, batch_size = 5, validation_data=(tm.X_test, tm.y_test), verbose=1)

#saving the model 
model.save('models/binaryChatbot_model6.h5', chatbotModel)
print("model created")

#this is to make a graph from the accuracy and loss of the chatbot model made
plt.plot(chatbotModel.history['accuracy'], label='accuracy')
plt.plot(chatbotModel.history['loss'], label = 'loss')
plt.title('model graph')
plt.show()
plt.savefig('graphs/modelLossVsAccuracy')


#accuracy vs validation accuracy
plt.plot(chatbotModel.history['accuracy'])
plt.plot(chatbotModel.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.savefig('graphs/modelAccuracy')



#loss vs validation loss
plt.plot(chatbotModel.history['loss'])
plt.plot(chatbotModel.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
plt.savefig('graphs/model loss')
