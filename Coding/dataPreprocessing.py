#DO DATA PREPROCESSING STEPS WHICH ARE: 
    #Removing punctuations like . , ! $( ) * % @
    #Removing URLs
    #Removing Stopwords
    #Lower casing
    #Tokenization
    #Stemming
    #Lemmatization


import ReadingDataFiles as dataFiles
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer 
#nltk.download("punkt")
#nltk.download('stopwords') 

############################################# GOT THIS CODE FROM SOMEWHERE  ######################################################
###############   NEED TO EDIT IT TO PREPROCESS THE DATA FROM MY CSV TRAINING FILE
#https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/

#DEFINING THE FUNCTION TO REMOVE PUNCTUATION AND TOKENIZE 
#
def remove_punctuation_tokenize(text):
    #removes the punctuations
    punctuationfree = nltk.RegexpTokenizer(r"\w+")
    #tokenizes sentence - splits it into smaller parts
    individual_words = punctuationfree.tokenize(text)
    #print (new_words)
    return individual_words

#storing the puntuation free text
dataFiles.concate_data['CleanQuestion']= dataFiles.concate_data['Question'].apply(lambda x:remove_punctuation_tokenize(str(x))) #i put str(x), cause it expects it back as a string unless it gives an error
dataFiles.concate_data['CleanAnswer']= dataFiles.concate_data['Answer'].apply(lambda x:remove_punctuation_tokenize(str(x)))



#DEFINING THE FUNTION TO REMOVE STOPWORDS FROM TOKENIZED TEXT
def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    output= [i for i in text if i not in stopwords]
    return output

#applying the function
#dataFiles['no_stopwords']= dataFiles['msg_tokenied'].apply(lambda x:remove_stopwords(x)) RemoveStopwordsQuestion
dataFiles.concate_data['StopWordsAnswer']= dataFiles.concate_data['CleanAnswer'].apply(lambda x:remove_stopwords(x))
print(dataFiles.concate_data['StopWordsAnswer'])


"""""

#importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer
#defining the object for stemming
porter_stemmer = PorterStemmer()

#defining a function for stemming
def stemming(text):
stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
dataFiles['msg_stemmed']=dataFiles['no_sw_msg'].apply(lambda x: stemming(x))



from nltk.stem import WordNetLemmatizer
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
#defining the function for lemmatization
def lemmatizer(text):
lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
dataFiles['msg_lemmatized']=dataFiles['no_stopwords'].apply(lambda x:lemmatizer(x))

"""