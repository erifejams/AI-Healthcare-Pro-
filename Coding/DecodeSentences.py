

import numpy as np
import TrainingModel as tm
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from pythainlp.util import normalize

#loading the modal
chatbotmodal = load_model('./models/binaryChatbot_model3.h5', compile = False)

"""
input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])
reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())


pred = bm.model.predict([X_train, y_train])

for i,j in zip(pred, y_train):
    print(np.argmax(i), np.argmax(j))
"""


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

    inputs = np.array(pad_sequences(encoder(inputs), maxlen=3120))
    
    return inputs

def messange_to_bot(sentences ):
    sentence = ''
    word = preprocessing([sentences])
    continued = True
    answer_index = [[]]
    while continued:
        x = np.array(pad_sequences(word, maxlen=3120))
        predict = np.argmax(chatbotmodal.predict([x]), axis=1)[0]
        #print(predict)
        if predict == 0:
            continued = False
            break
        
        answer_index[0].append(predict)

        if predict != 0:
            for i in tm.question_index_word:
                if predict == i:
                    sentence += tm.question_index_word[i]
                    break
                else:
                    #print(sentence)
                    break
                    #sentence += tm.question_index_word[predict]
    return sentence

text = "hello"
print(messange_to_bot(text))
"""
def talk_with_bot():
    while True:
        text = input('You : ')
        if text == 'quit':
            break
        print('InTa :', messange_to_bot(text))

talk_with_bot()

"""