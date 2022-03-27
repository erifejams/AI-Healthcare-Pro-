import os

from sklearn import preprocessing

#have to include this or some libraries(in tensorflow) are not found before importing tensorflow
#apparently this problem was only found in Python 3.9.10 not python
#MAKE SURE TO ALWAYS INCLUDE IN THE FILE **IMPORTANT
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
import TrainingModel as tm
import dataPreprocessing as dt


#loading the modal
chatbotmodal = load_model('./chatbot_model.h5', compile = True)
#firstline 
print("InTa:  Hi!!!, my name is Inta, nice to meet you!!!")

#getting the pridicted response from the chatbot
while True:
    texts_p = []
    prediction_input = input('You: ')
    if( prediction_input != "goodbye"):
        if(prediction_input == "thanks" or prediction_input == "thank you"):
            print("InTa: it's no problem")
        else:

            #tokenizing and padding
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(prediction_input)
            prediction_input = tokenizer.texts_to_sequences(prediction_input)
            prediction_input = pad_sequences(prediction_input, padding='post')
            prediction_input = np.array(prediction_input).reshape(-1,1)
            print(prediction_input)

            label_enc = preprocessing.LabelEncoder()
            #fit model
            label_enc.fit(tm.X_train, tm.y_train.ravel())

            #predicting the response
            #output = label_enc.predict([prediction_input])[0]
            output = label_enc.predict([prediction_input])[0]
            print(output)
            output = np.argmax(output, 1)
            #print(tm.training[int(output[0][0])])

            #finding the right response 
            #response_tag = label_enc.inverse_transform(output)
            print("InTa: ", int(output))

    if(prediction_input == "goodbye"):
        print("InTa: bye:)")
        break