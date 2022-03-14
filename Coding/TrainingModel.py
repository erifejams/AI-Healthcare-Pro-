#CREATE TRAINING AND TEST LIST
#Tokenizing the training data


import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#THIS IS TO READ THE TRAINING DATA
training = pd.read_csv (r'Data/trainingData.csv')


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

# Marathi Word --> index dict
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