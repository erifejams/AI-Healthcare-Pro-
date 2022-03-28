#CREATE TRAINING AND TEST LIST
#Tokenizing the training data

import os
#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#THIS IS TO READ THE TRAINING DATA
training = pd.read_csv (r'Data/trainingData.csv')

#print(training['Answer'])
# this converts into a list
question_list = training.Question.to_list()
answer_list = training.Answer.to_list()

#this turns the words into numbers input, so that it can be used for training the model
#input was taken as list
def tokenize_sent(text):

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(text)

  #returning the tokenizer and enocoded text
  return tokenizer, tokenizer.texts_to_sequences(text)


# Tokenize question and answer sentences
question_tokenizer, question_encoded = tokenize_sent(text = question_list)
answer_tokenizer, answer_encoded = tokenize_sent(text = answer_list)

# question Word --> index dictionary
question_index_word = question_tokenizer.index_word

# question Index --> word dictionary
question_word_index = question_tokenizer.word_index
#print(question_word_index)

# size of question vocabulary for encoder input
# For zero padding we have to add +1 in size
QUESTION_VOCAB_SIZE = len(question_tokenizer.word_counts)+1 #18976

# answer Word --> index dict
answer_word_index= answer_tokenizer.word_index 
#answer vocab size for decoder output
ANSWER_VOCAB_SIZE = len(answer_tokenizer.word_counts)+1 #22201

# Getting max length of question and answer sentences
max_question_len = 0
for i in range(len(question_encoded)):
  if len(question_encoded[i]) > max_question_len:
    max_question_len = len(question_encoded[i])
#print(max_question_len) #3120


max_answer_len = 0
for i in range(len(answer_encoded)):
  if len(question_encoded[i]) > max_answer_len:
    max_answer_len= len(answer_encoded[i])
#print(max_answer_len) #585


# this pads the question and the response
#to make sure all the sequences have the same length, if not zero is added unto the end
question_padded = pad_sequences(question_encoded, maxlen = 3120, padding='post')
answer_padded = pad_sequences(answer_encoded, maxlen =  585, padding='post')

#print(question_padded)
#print(answer_padded)

#neural network is best in range 0-1, so min-max is better to user
mms = MinMaxScaler()
question_padded = mms.fit_transform(question_padded)
answer_padded = mms.fit_transform(answer_padded)


# Split data into train and test set, 70-30 split ratio
X_train, X_test, y_train, y_test = train_test_split(question_padded, answer_padded, test_size=0.3, random_state=1)
#print(len(y_train[0]))
#(12581, 3120)
#print(X_train.shape)
#(5392, 3120)
#print(X_test.shape)
#(12581, 585)
#print(y_train.shape)
#(5392, 585)
#print(y_test.shape)

