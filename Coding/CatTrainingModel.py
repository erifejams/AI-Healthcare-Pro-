#CREATE TRAINING AND TEST LIST
#Tokenizing the training data

import os
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
from keras.optimizers import gradient_descent_v2


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
question_input  = le.fit_transform( training['Question'])
answer_output = le.fit_transform(training['Answer'])

#question_input = question_input.str.replace(',', '')
#question_input = question_input.str.replace("' '", ' ')
#neural network is best in range 0-1, so min-max is better to user
#scaling data
mms = MinMaxScaler()
question_padded = mms.fit_transform(question_input.reshape(len(question_input), 1))
answer_padded = mms.fit_transform(answer_output.reshape(len(answer_output), 1))
print(answer_padded[3])

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(question_input, answer_output, test_size=0.33, random_state=1)


# make output 3d
#y_train_enc = y_train_enc.values.reshape((len(y_train_enc), 1, 1))
#y_test_enc = y_test_enc.values.reshape((len(y_test_enc), 1, 1))

# summarize
#print('Train', X_train.shape, y_train.shape) #Train (12041,) (12041, 1)
#print('Test', X_test.shape, y_test.shape) #Test (5932,) (5932, 1)

#import classifier algorithm here
from sklearn.linear_model import LogisticRegression

# create classifier
lg_model = LogisticRegression()

#Training the classifier
lg_model.fit(X_train.reshape(-1, 1),y_train)

# import evaluation metrics
from sklearn.metrics import confusion_matrix, accuracy_score

# evaluate the model
y_pred = lg_model.predict(X_test)
print(y_pred)

# Get the accuracy
print("Accuracy Score of Logistic Regression classifier: ","{:.4f}".format(accuracy_score(y_test, y_pred)))