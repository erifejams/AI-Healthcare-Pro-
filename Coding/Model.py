#CREATING A MODEL USING SEQ2SEQ NEURAL NETWORK (FRAMEWORK )

###### SOURCE WHERE I AM GETTING THE CODE HELP FROM 
#https://github.com/prasoons075/Deep-Learning-Codes/blob/master/Encoder%20Decoder%20Model/Encoder_decoder_model.ipynb
#https://towardsdatascience.com/generative-chatbots-using-the-seq2seq-model-d411c8738ab5


import os

#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


import tensorflow as tf
from keras.utils.vis_utils import plot_model
import TrainingModel as tm
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Concatenate
from keras import Input, Model
from keras.layers.attention import Attention

# configure
num_encoder_tokens = 71
num_decoder_tokens = 93
latent_dim = 256

# Encoder input
encoder_inputs = Input(shape=(None,)) 

# Embedding layer- i am using 1024 output-dim for embedding you can try diff values 100,256,512,1000
enc_emb = Embedding(tm.QUESTION_VOCAB_SIZE, 1024)(encoder_inputs)

# Bidirectional lstm layer
enc_lstm1 = Bidirectional(LSTM(256,return_sequences=True,return_state=True))
encoder_outputs1, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm1(enc_emb)

# Concatenate both h and c 
final_enc_h = Concatenate()([forw_state_h,back_state_h])
final_enc_c = Concatenate()([forw_state_c,back_state_c])

# get Context vector
encoder_states =[final_enc_h, final_enc_c]


#  decoder input
decoder_inputs = Input(shape=(None,)) 

# decoder embedding with same number as encoder embedding
dec_emb_layer = Embedding(tm.ANSWER_VOCAB_SIZE, 1024) 
dec_emb = dec_emb_layer(decoder_inputs)   # apply this way because we need embedding layer for prediction 

# In encoder we used Bidirectional so it's having two LSTM's so we have to take double units(256*2=512) for single decoder lstm
# LSTM using encoder's final states as initial state
decoder_lstm = LSTM(512, return_sequences=True, return_state=True) 
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

#REMOVED THE ATTENTION LAYER BECAUSE I COULDN'T IMPORT IT



# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 




# compile model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("give Your path to save check points", monitor='val_accuracy')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
callbacks_list = [checkpoint, early_stopping]

# Training set
encoder_input_data = tm.X_train
# To make same as target data skip last number which is just padding
decoder_input_data = tm.y_train[:,:-1]
# Decoder target data has to be one step ahead so we are taking from 1 as told in keras docs
decoder_target_data =  tm.y_train[:,1:]

# devlopment set
encoder_input_test = tm.X_test
decoder_input_test = tm.y_test[:,:-1]
decoder_target_test=  tm.y_test[:,1:]

EPOCHS= 20 #@param {type:'slider',min:10,max:100, step:10 }
history = model.fit([encoder_input_data, decoder_input_data],decoder_target_data, 
                    epochs=EPOCHS, 
                    batch_size=5,
                    validation_data = ([encoder_input_test, decoder_input_test],decoder_target_test),
                    callbacks= callbacks_list)

# Don't forget to save weights of trained model 
model.save_weights("model.h5") # can give whole path to save model


"""
# Define an input sequence and process it.
encoder_inputs = Input(shape=(tm.question_padded, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(tm.answer_padded, num_decoder_tokens))

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
# define encoder inference model
encoder_model = Model(encoder_inputs, encoder_states)


# define decoder inference model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# summarize model
#pydot.call_graphviz.plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True)
#plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)
"""