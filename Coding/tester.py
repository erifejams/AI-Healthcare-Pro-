import os
import pandas as pd

#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from tensorflow import optimizers
from keras.layers import Dense, Dropout, Normalization
from keras.models import Sequential, Model 

#THIS IS TO READ THE TRAINING DATA
training = pd.read_csv (r'Data/trainingData.csv')
# load the dataset as a pandas DataFrame
data = training
# retrieve numpy array
dataset = data


# split into input (X) and output (y) variables
X = dataset['Question']
y = dataset['Answer']
print(y)

# format all fields as floats
X = X.astype(np.array)
# reshape the output variable to be one column (e.g. a 2D shape)
y = y.reshape((len(y), 1))


# prepare input data using min/max scaler.
def prepare_inputs(X_train, X_test):
    oe = MinMaxScaler()
    X_train_enc = oe.fit_transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc

# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    ohe = OneHotEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    y_train_enc = ohe.fit_transform(y_train).toarray()
    y_test_enc = ohe.transform(y_test).toarray()
    return y_train_enc, y_test_enc



X, y = load_dataset("csv_ready.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

#prepare_input function missing here
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
print('Finished preparing inputs.')

# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation="relu")) 
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

#opt = optimizers.Adam(lr=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train_enc, y_train_enc, epochs=20, batch_size=32, verbose=1, use_multiprocessing=True)

_, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
print('Accuracy: %.2f' % (accuracy * 100))