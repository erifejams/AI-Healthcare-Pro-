import os

#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


import tensorflow as tf
from tensorflow import keras as ks
from keras.utils.vis_utils import plot_model
import TrainingModel as tm
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Concatenate
from keras import Input, Model, layers
from keras.layers.attention import Attention
import TrainingModel as tm
from keras.models import load_model


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


model = ks.Sequential()
model.add(layers.Embedding(input_length = 3120, input_dim=10000, output_dim=3120))    
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.LSTM(64))
model.add(layers.Dense(585, activation="softmax"))
model.compile(optimizer=ks.optimizers.SGD(learning_rate=(0.55)), loss=ks.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

#TF_CUDNN_RESET_RND_GEN_STATE=1
model.fit(tm.X_train, tm.y_train, batch_size=5, epochs=25)


#model.save("rnn")


