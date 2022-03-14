#CREATING A MODEL USING SEQ2SEQ NEURAL NETWORK (FRAMEWORK )

###### SOURCE WHERE I AM GETTING THE CODE HELP FROM 
#https://github.com/prasoons075/Deep-Learning-Codes/blob/master/Encoder%20Decoder%20Model/Encoder_decoder_model.ipynb
#https://towardsdatascience.com/generative-chatbots-using-the-seq2seq-model-d411c8738ab5

import os
#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


from json.encoder import py_encode_basestring
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils.vis_utils import plot_model

# configure
num_encoder_tokens = 71
num_decoder_tokens = 93
latent_dim = 256


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# plot the model
#plot_model(model, to_file='model.png', show_shapes=True)