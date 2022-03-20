
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
import numpy as np

batch_size = 5  # Batch size for training.
epochs = 5  # Number of epochs to train for.
latent_dim = 525  # Latent dimensionality of the encoding space.
num_samples = 17930  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'Data/trainingData.csv'#'fra-eng/fra.txt'

num_encoder_tokens = 71
num_decoder_tokens = 93
latent_dim = 25

input_token_index = dict(
    [(char, i) for i, char in enumerate(tm.question_padded)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(tm.answer_padded)])

encoder_input_data = np.zeros(
    (len(tm.question_list), tm.max_question_len, num_encoder_tokens))
decoder_input_data = np.zeros(
    (len(tm.question_list), tm.max_answer_len, num_decoder_tokens))
decoder_target_data = np.zeros(
    (len(tm.question_list), tm.max_answer_len, num_decoder_tokens))

print("encoder_input_data",encoder_input_data.shape)
print("decoder_input_data",decoder_input_data.shape)
print("decoder_target_data",decoder_target_data.shape)
print("**************")