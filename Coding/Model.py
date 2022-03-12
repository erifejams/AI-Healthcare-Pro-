#CREATING A MODEL USING SEQ2SEQ NEURAL NETWORK (FRAMEWORK )

###### SOURCE WHERE I AM GETTING THE CODE HELP FROM 
#https://github.com/prasoons075/Deep-Learning-Codes/blob/master/Encoder%20Decoder%20Model/Encoder_decoder_model.ipynb
#https://towardsdatascience.com/generative-chatbots-using-the-seq2seq-model-d411c8738ab5


import TrainingModel as tm
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical