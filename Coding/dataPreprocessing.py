#DATA PREPROCESSING STEPS WHICH ARE: 
    #Removing punctuations like . , ! $( ) * % @ AND Tokenization
    #Removing URLs
    #Removing Stopwords - MAY DO THIS STEP LATER, BUT I THOUGHT I SHOULDN'T REMOVE OR IT WILL CHANGE THE MEANING OF THE SENTENCE
    #Lower casing - DIDN'T DO THIS STEP AS IT COULD CHANGE THE MEANING OF THE SENTENCE
    #Stemming - LESS ACCURATE RESULTS, SO FOR NORMALIZATION OF THE TEXT WENT WITH LEMMITZATION
    #Lemmatization


import ReadingDataFiles as dataFiles
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
#nltk.download("punkt")
#nltk.download('stopwords') 
#nltk.download('omw-1.4')

#############################################  ######################################################
###############   NEED TO EDIT IT TO PREPROCESS THE DATA FROM MY CSV TRAINING FILE
#https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/

#PUNCTUATION AND TOKENIZATION STEP
#DEFINING THE FUNCTION TO REMOVE PUNCTUATION AND TOKENIZE 
def remove_punctuation_tokenize(text):
    #removes the punctuations
    punctuationfree = nltk.RegexpTokenizer(r"\w+")
    #tokenizes sentence - splits it into smaller parts
    individual_words = punctuationfree.tokenize(text)
    #print (new_words)
    return individual_words

#NOT REMOVING STOPWORDS AS IT COULD CHANGE THE MEANING WHEN DOING SENTIMENT ANALYSIS

#POS TAGGING STEP - CATOGIRISING IN VERBS, NOUN, 
def pos_tagging(text):
    tagging = nltk.pos_tag(text)
    return tagging


#LEMMATIZATION STEP
#lemmatization provides better results, but there might be trouble with speed, if there is go with stemming
wordnet_lemmatizer = WordNetLemmatizer()
#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = wordnet_lemmatizer.lemmatize(text) 
    return str(lemm_text)


#VECTORIZED STEP


#storing the puntuation and tokenized text 
dataFiles.concate_data['CleanQuestion']= dataFiles.concate_data['Question'].apply(lambda x:remove_punctuation_tokenize(str(x))) #i put str(x), cause it expects it back as a string unless it gives an error
dataFiles.concate_data['CleanAnswer']= dataFiles.concate_data['Answer'].apply(lambda x:remove_punctuation_tokenize(str(x)))

#storing the pos tags
dataFiles.concate_data['tagginQuestion']= dataFiles.concate_data['CleanQuestion'].apply(lambda x:pos_tagging(x))
dataFiles.concate_data['tagginAnswer']= dataFiles.concate_data['CleanAnswer'].apply(lambda x:pos_tagging(x))

#storing the Lemmatization
dataFiles.concate_data['AnswerLemmatized']= dataFiles.concate_data['tagginAnswer'].apply(lambda x:lemmatizer(str(x)))
dataFiles.concate_data['QuestionLemmatized']= dataFiles.concate_data['tagginQuestion'].apply(lambda x:lemmatizer(str(x)))

#REMOVING COLUMNS NOT NEEDED AGAIN
dataFiles.concate_data.drop(['Question', 'Answer' ,'CleanQuestion', 'CleanAnswer', 'tagginQuestion', 'tagginAnswer'], axis = 1, inplace = True)

dataFiles.concate_data.to_csv('Data/trainingData.csv')
#print(dataFiles.concate_data['QuestionLemmatized'])