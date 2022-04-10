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
chatbotmodal = load_model('./models/chatbot_model.h5', compile = True)
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
            prediction_input = pad_sequences(prediction_input, padding='post')
            prediction_input = np.array(prediction_input).reshape(-1,1)
            print(str(prediction_input))

            label_enc = preprocessing.LabelEncoder()
            #FIT MODEL
            label_enc.fit(chatbotmodal)


            #PREDICTING THE RESPONSE
            #output = label_enc.predict([prediction_input])[0]
            output = label_enc.predict(np.array([prediction_input]))[0]
            print(output)
            #to get the highest posibility
            output_index = np.argmax(output[0, -1, :])
            print(tm.training[int(output[0][0])])

            """
            if output[output_index] > 0.70:
                #finding the right response 
                #response_tag = label_enc.inverse_transform(output)
                print("InTa: ", str(output))
            else:
                print("I don’t fully understand")


            def chat_with_bot():
            checkpoint = './checkpoint.ckpt'
            session = tf.InteractiveSession()
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(session, checkpoint)
            while True:
            question = input("You: ")
            if(question == 'Goodbye'):
                break
            else:
                question = convert_string2int(question, questionswords2int)
                question = question + [questionswords2int['<PAD>']] * (20 - len(question))
                fake_batch = np.zeros((batch_size, 20))
                fake_batch[0] = question
                predicted_answer = session.run(test_predictions, {input: fake_batch, keep_prob: 0.5})[0]
                answer = ''
                for i in np.argmax(predicted_answer, 1):
                    if answersints2word[i] == 'i':
                        token = 'I'
                    elif answersints2word[i] == '<EOS>':
                        token = '.'
                    elif answersints2word[i] == '<OUT>':
                        token = 'out'
                    else:
                        token = ' ' + answersints2word[i]
                    answer += token
                    if token == '.':
                        break
                print('Chatbot:', answer)
            """

            #IF THERE IS NO PREDICTION THAT IT CAN COME UP WITH 
            if prediction_input == 0:
                print("InTa: I don't understand, more like I'm not sure what to say to that?")
               

    if(prediction_input == "goodbye"):
        print("InTa: bye:)")
        break

    

