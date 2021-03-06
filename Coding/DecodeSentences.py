##THIS IS THE CHATBOT TALKING FILE
#IT INCLUDES THE CODE FOR THE DATABASE, PROCESSING USER INPUT, CHATBOT CHATTING
import os

from sklearn.preprocessing import MinMaxScaler

#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

#pip install pythainlp
#pip install keras

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from pythainlp.util import normalize
import TrainingModel as tm

#DATABASE IMPORTS
import sqlite3
from textblob import TextBlob
from unidecode import unidecode

#loading the modal
chatbotmodal = load_model('./models/binaryChatbot_model5.h5', compile = False)


#creating a database
#CREATING A DATABASE FOR USER INPUT WHILE THE USER IS WRITING THEIR SENTENCES
databaseUser = sqlite3.connect('Database/UserTable.db')
cursor = databaseUser.cursor()

#Once table is created comment it out
cursor.execute("""CREATE TABLE IF NOT EXISTS User (sentence TEXT, sentiment REAL)""")


word2index = {}
index2word = {}
word2index[' '] = 0
index2word[0] = ' '

def add_word(word):
    if not(word in word2index.keys()):
        current_index = len(word2index)
        word2index[word] = current_index
        index2word[current_index] = word


def encoder(input_sentences):
    input_sentences = input_sentences.copy()
    list_sentence = []
    for sentence in input_sentences:
        inputs = []
        for i in sentence:
            inputs.append(word2index[i])
        inputs = np.array(inputs)
        list_sentence.append(inputs)
    return list_sentence


def preprocessing(inputs):
    inputs = inputs.copy()
    for i, words in enumerate(inputs):
        word = normalize(words)
        word = word_tokenize(word)
        for j, w in enumerate(word):
            word[j] = correct(w)
            add_word(word[j])
        inputs[i] = word

    inputs = pad_sequences(encoder(inputs), maxlen=3120, padding='post')
    mms = MinMaxScaler()
    inputs = mms.fit_transform(inputs)    
    return inputs

def messange_to_bot(sentences):
    sentence = " "
    ij = 0
    word = preprocessing([sentences])
    continued = True

    while continued:
        #argmax identifies the maximum value in the prediction
        predict = np.argmax(chatbotmodal.predict(word), axis=1)[0] #the 0 removes the [] around the prediction
        #print(predict)
        if predict == 0:
            sentence = "I don't understand, more like I'm not sure what to say to that?"
            continued = False
            break

        #that means a predict exists, so its more than 0
        if predict > 0:
            #to check if the question asked by the user is in the file, 
            #to check if the answer given by the user is in the file, 
            for i in range(len(tm.AA)):
                questionI = tm.AT[i]
                answerI = tm.AA[i]

                if sentences == answerI: #if a response is given the question on the next line will be asked
                    #if so, then prints the answer
                    sentence = tm.AT[ij + 1]
                    print(ij)
                    #print(sentence)
                    continued = False
                    break

                if sentences == questionI: ## if a question is asked an answer will be given
                    sentence = tm.AA[ij]
                    print(ij)
                    #print(sentence)
                    continued = False
                    break

                else:
                    #it is 17972, cause of the amound of total sentences in training file
                    #has not yet finish going through all the sentences in training file, so it continues
                    if ij != 17973:
                        ij = ij + 1
                        continued = True
                    else:
                        #gone through all the sentences in training file but couldn't find matching
                        sentence = "I'm not sure what to say to that?"
                        continued = False  
                        break            
    return sentence


chatbot_name = "InTa"
def talk_with_bot(msg):

    conversationInput = unidecode(msg.lower())
    #CHECKS THE SENTIMENT ANALYSIS OF THE SENTENCE
    analysis = TextBlob(conversationInput)
    sentiment = analysis.sentiment.polarity
    #print(conversationInput, sentiment)

    #WRITE TO THE DATABASE AND INSERT INTO USER TABLE
    cursor.execute("INSERT INTO User (sentence, sentiment) VALUES ('{}', '{}')".format(conversationInput, sentiment))


    #prints out reply from chabot
    #print(index2word)
    return messange_to_bot(conversationInput)
