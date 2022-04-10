

import numpy as np
import pandas as pd
import TrainingModel as tm
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from pythainlp.util import normalize

#loading the modal
chatbotmodal = load_model('./models/binaryChatbot_model4.h5', compile = False)

training = pd.read_csv (r'Data/trainingData.csv')

#from pythainlp import word_vector
# model = word_vector.get_model()
#thai2dict = {}
# for word in tm.question_word_index:
#     print(word, model[word])
#     thai2dict[word] = model[word]
# thai2vec = pd.DataFrame.from_dict(thai2dict,orient='index')
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

    inputs = pad_sequences(encoder(inputs), maxlen=3120)
    return inputs

def messange_to_bot(sentences):
    sentence = " "
    ij = 0
    word = preprocessing([sentences])
    continued = True
    while continued:
        x = np.array(pad_sequences(word, maxlen=3120))
        #argmax identifies the maximum value in the prediction
        predict = np.argmax(chatbotmodal.predict(x), axis=1)[0] #the 0 removes the [] around the prediction
        if predict == 0:
            continued = False
            break

        for i in tm.AT:
            if sentences == i:
                #to check if the question asked by the user is in the file, if so, then prints the answer
                sentence = tm.AA[ij-1]
                continued = False
                break
            else:
                ij = ij + 1
                continued = True
    return sentence


#text = "I'm feeling gloomy"
#print(messange_to_bot(text))


def talk_with_bot():
    #print('InTa : Hi, my name is InTa, you can just say bye when you want to stop talking to me')
    while True:
        text = input('You : ')
        if text == 'quit':
            break
        print('InTa :', messange_to_bot(text))

talk_with_bot()
