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
import nltk
from nltk.stem import WordNetLemmatizer 


wordnet_lemmatizer = WordNetLemmatizer()
def clean_up(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[ wordnet_lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


#loading the modal
chatbotmodal = load_model('./models/binaryChatbot_model5.h5', compile = True)
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

            #TOKENIZING AND PADDING 
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(prediction_input)
            prediction_input = tokenizer.texts_to_sequences(prediction_input)
            prediction_input = pad_sequences(prediction_input, maxlen =3120, padding='post')
            mms = preprocessing.MinMaxScaler()
            prediction_input = mms.fit_transform(prediction_input)
            prediction_input = prediction_input.reshape(len(prediction_input),-1)


            #PREDICTING THE RESPONSE
            #output = label_enc.predict([prediction_input])[0]
            output = chatbotmodal.predict([prediction_input])[0]
            #to get the highest posibility
            output_index = np.argmax(output[0])
            
         
            #finding the right response 
            for i in tm.question_index_word:
                if i == output:
                    response_tag = tm.question_index_word[i]
                    print("InTa: ", str(response_tag))

            #IF THERE IS NO PREDICTION THAT IT CAN COME UP WITH 
            if prediction_input == 0:
                print("InTa: I don't understand, more like I'm not sure what to say to that?")
               

    if(prediction_input == "goodbye"):
        print("InTa: bye:)")
        break

    

