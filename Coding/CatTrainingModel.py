#CREATE TRAINING AND TEST LIST
#Tokenizing the training data

import os

from sklearn.linear_model import SGDOneClassSVM
#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt #to plot the graph


#THIS IS TO READ THE TRAINING DATA
training = pd.read_csv (r'Data/trainingData.csv')
#split data into answer and reponse
question_input2 = training['Question']
#answer_output = training.Answer

#question_input = training['Question'].to_list()
#answer_output =training['Answer'].to_list()

# format all fields as string

#question_input =  question_input.values.reshape(-1,1)
#question_input = question_input.astype(str)
#extracts a numpy array with the values of your pandas Series object and then reshapes it to a 2D array - was given 1d from data before changed to 2d
#answer_output =  answer_output.values.reshape(-1,1)
#answer_output = answer_output.values.reshape((len(answer_output), 1))


# Convert target label to numerical Data
le = LabelEncoder()
question_input  = le.fit_transform( training['Question'].to_list())
answer_output = le.fit_transform(training['Answer'].to_list())

#question_input = question_input.str.replace(',', '')
#question_input = question_input.str.replace("' '", ' ')
#neural network is best in range 0-1, so min-max is better to user
#scaling data
mms = MinMaxScaler()
question_padded = mms.fit_transform(question_input.reshape(len(question_input),1))
answer_padded = mms.fit_transform(answer_output.reshape((len(answer_output), 1)))


# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(question_padded, answer_padded, test_size=0.33, random_state=1)



import numpy as np
import TrainingModel as tm

from keras.models import Sequential
from keras.layers import Dense, Dropout



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
model.add(Dense(len(tm.y_train[0]), activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])

#fitting the model
#chatbotModel = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size = 8, validation_data=(tm.X_test, tm.y_test), verbose=1)
#to make it go process faster, it it turned into a numpy array
chatbotModel = model.fit(np.array(tm.X_train), np.array(tm.y_train), epochs=5, batch_size = 5, validation_data=(tm.X_test, tm.y_test), verbose=1)

#saving the model 
model.save('models/binaryChatbot_model6.h5', chatbotModel)
print("model created")
