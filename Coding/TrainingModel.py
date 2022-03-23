#CREATE TRAINING AND TEST LIST
#Tokenizing the training data
import os
#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

from matplotlib import pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Concatenate, Attention, TimeDistributed
from keras import Input, Model


import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#THIS IS TO READ THE TRAINING DATA
training = pd.read_csv (r'Data/trainingData.csv')

training['Answer'] = training.Answer.apply(lambda x: 'start '+ x + ' stop')

#print(training['Answer'])
# Convert into list of sentence we need list to pass in tokenizer
question_list = training.Question.to_list()
answer_list = training.Answer.to_list()

def tokenize_sent(text):
  '''
  Take list on texts as input and 
  returns its tokenizer and enocoded text
  '''
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(text)

  return tokenizer, tokenizer.texts_to_sequences(text)


# Tokenize question and answer sentences
question_tokenizer, question_encoded = tokenize_sent(text = question_list)
answer_tokenizer, answer_encoded = tokenize_sent(text = answer_list)

# question Word --> index dictionary
question_index_word = question_tokenizer.index_word

# question Index --> word dictionary
question_word_index = question_tokenizer.word_index

# size of question vocabulary for encoder input
# For zero padding we have to add +1 in size
QUESTION_VOCAB_SIZE = len(question_tokenizer.word_counts)+1

# answer Word --> index dict
answer_word_index= answer_tokenizer.word_index
#answer vocab size for decoder output
ANSWER_VOCAB_SIZE = len(answer_tokenizer.word_counts)+1

# Getting max length of question and answer sentences
max_question_len = 0
for i in range(len(question_encoded)):
  if len(question_encoded[i]) > max_question_len:
    max_question_len= len(question_encoded[i])

max_answer_len = 0
for i in range(len(answer_encoded)):
  if len(question_encoded[i]) > max_answer_len:
    max_answer_len= len(answer_encoded[i])

# Padding both
question_padded = pad_sequences(question_encoded, maxlen = max_question_len, padding='post')
answer_padded = pad_sequences(answer_encoded, maxlen = max_answer_len, padding='post')

# Convert to array
question_padded= np.array(question_padded)
answer_padded= np.array(answer_padded)

#print(question_padded)
#print(answer_padded)


# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(question_padded, answer_padded, test_size=0.1, random_state=0)

"""
#SHUFFLE DATA - THIS IS TO MAKE SURE WE HAVE RANDOM SAMPLE 
random.shuffle(list(training))
training = np.array(training)

#create train and test lists. X - Question, Y - Anwser
#the O was the index id
train_x = list(training[:, 1])
train_y = list(training[:, 2])
#print(training)

"""
"""
# Encoder input
encoder_inputs = Input(shape=(max_question_len)) 

# Embedding layer- i am using 1024 output-dim for embedding you can try diff values 100,256,512,1000
enc_emb = Embedding(QUESTION_VOCAB_SIZE, 1024)(encoder_inputs)

# Bidirectional lstm layer
enc_lstm1 = Bidirectional(LSTM(512,return_sequences=True,return_state=True))
##encoder_outputs1, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm1(enc_emb)
encoder_output, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm1(enc_emb)


# get Context vector
encoder_states =[forw_state_h, forw_state_c, back_state_h, back_state_c]


#  decoder input
decoder_inputs = Input(shape=(max_answer_len)) 

# decoder embedding with same number as encoder embedding
dec_emb = Embedding(ANSWER_VOCAB_SIZE, 1024)(decoder_inputs) 
# apply this way because we need embedding layer for prediction 

# In encoder we used Bidirectional so it's having two LSTM's so we have to take double units(256*2=512) for single decoder lstm
# LSTM using encoder's final states as initial state
decoder_lstm = LSTM(512, return_sequences=True, return_state=True) 
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = encoder_states)

# Using Attention Layer
#attention_layer = AttentionLayer()
attention_layer = Attention()
attention_result, attention_weights = attention_layer([encoder_output, decoder_outputs])

# Concat attention output and decoder LSTM output 
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])

# Dense layer with softmax
decoder_dense = Dense(ANSWER_VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat_input)

attention_result, attention_weights = attention_layer([encoder_output, decoder_outputs])
dense1 = Dense(20, activation="relu")(attention_result)
output = Dense(1, activation="sigmoid")

# Define the model
model = Model(inputs = [encoder_inputs, decoder_inputs], outputs = decoder_outputs)

model.summary()
"""
