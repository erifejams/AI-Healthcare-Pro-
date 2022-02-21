#I WANT TO PUT ALL THE DATA INTO ONE CSV FILE AS ID, QUESTION, ANSWER/ Response
#DO DATA PREPROCESSING STEPS WHICH ARE: 
    #Removing punctuations like . , ! $( ) * % @
    #Removing URLs
    #Removing Stop words
    #Lower casing
    #Tokenization
    #Stemming
    #Lemmatization

import pandas as pd
import string

#########################################################  MY CODE   ###########################################################
#reading the data
#https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/
data = pd.read_csv('Data/Psych_data.csv', encoding="ISO-8859-1")
#putting only the column that I needed from Psych_data.csv
data = data[['Answer', 'Question']]
#this is to read the data to a csv file
data.to_csv('Data/trainingData.csv')


#turn txt file to a csv file
#then put it under a column
#then find the word e.g human 1 and put it under a 2 separate columns e.g Question and answer
firstConv = []
secondConv = []
with open('Data/human_chat.txt', encoding="utf8") as file:
    NormalConv = file.readlines()
    
    #TO SEPARTE IT BY HUMAN 1 AND HUMAN 2
    for i in NormalConv:   
        #QUESTION KINDA SENTENCES
        if "Human 2" in i:
            firstConv.append(i) 
            #print(data)
        #ANSWERS KINDA SENTENCES
        if "Human 1" in i:
            secondConv.append(i)

#df = pd.DataFrame(firstConv, columns = ['Question', 'Answer'])

#PUT IT INTO THE TRAINING FILE
# print(firstConv)


#qna_chitchat_professional.tsv
data3 = pd.read_csv("Data/qna_chitchat_professional.tsv", sep='\t')
data3 = data3[['Answer', 'Question']]
data3.to_csv('Data/trainingData.csv')

print(data3)
#https://www.geeksforgeeks.org/different-ways-to-create-pandas-dataframe/
#df = pd.DataFrame(data, columns = ['Question', 'Answer'])



############################################# GOT THIS CODE FROM SOMEWHERE  ######################################################
###############   NEED TO EDIT IT TO PREPROCESS THE DATA FROM MY CSV TRAINING FILE
"""
#defining the function to remove punctuation
string.punctuation

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
data['Question']= data['Question'].apply(lambda x:remove_punctuation(x))
data



#defining function for tokenization
import re
def tokenization(text):
    tokens = re.split('W+',text)
    return tokens
#applying function to the column
data['msg_tokenied']= data['msg_lower'].apply(lambda x: tokenization(x))



#importing nlp library
import nltk
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
stopwords[0:10]
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#applying the function
data['no_stopwords']= data['msg_tokenied'].apply(lambda x:remove_stopwords(x))




#importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer
#defining the object for stemming
porter_stemmer = PorterStemmer()

#defining a function for stemming
def stemming(text):
stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
data['msg_stemmed']=data['no_sw_msg'].apply(lambda x: stemming(x))



from nltk.stem import WordNetLemmatizer
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
#defining the function for lemmatization
def lemmatizer(text):
lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
data['msg_lemmatized']=data['no_stopwords'].apply(lambda x:lemmatizer(x))

"""