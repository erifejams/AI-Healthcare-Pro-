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
#READING THE DATA

#Psych_data.csv
data = pd.read_csv('Data/Original/Psych_data.csv', encoding="ISO-8859-1")
#putting only the column that I needed from Psych_data.csv
data = data[['Answer', 'Question']]


#human_chat.txt
#then find the word e.g human 1 and put it under a 2 separate columns e.g Question and answer
firstConv = []
secondConv = []
data2 = pd.DataFrame(columns = ['Question','Answer'])
with open('Data/Original/human_chat.txt', encoding="utf8") as file:
    NormalConv = file.readlines()
    
    #TO SEPARTE IT BY HUMAN 1 AND HUMAN 2
    for i in NormalConv:   
        #QUESTION KINDA SENTENCES
        if "Human 2" in i:
            firstConv.append(i) 
        #ANSWER KINDA SENTENCES
        if "Human 1" in i:
            secondConv.append(i)
            

data2 = pd.DataFrame(firstConv , columns = ['Question'])
#BECAUSE THE LENGTH OF VALUES DIDN'T MATCH(ONE WAS MORE THAN THE OTHER), SO USING SERIES NULL WAS FILLED IN
data2['Answer'] = pd.Series(secondConv)
data2.replace("Human 2:",'',inplace=True, regex=True)
data2.replace("Human 1:",'',inplace=True, regex=True)
data2.replace("\n",'',inplace=True, regex=True)


#qna_chitchat_professional.tsv
data3 = pd.read_csv("Data/Original/qna_chitchat_professional.tsv", sep='\t')
data3 = data3[['Answer', 'Question']]


#20200325_counsel_chat.csv
data4 = pd.read_csv('Data/Original/20200325_counsel_chat.csv', encoding="ISO-8859-1")
data4 = data4[['answerText', 'questionText']]
data4.rename(columns = {'answerText':'Answer'}, inplace = True)
data4.rename(columns = {'questionText':'Question'}, inplace = True)


#counselchat-data.csv
data5 = pd.read_csv('Data/Original/counselchat-data.csv', encoding="ISO-8859-1")
data5 = data5[['answerText', 'questionText']]
data5.rename(columns = {'answerText':'Answer'}, inplace = True)
data5.rename(columns = {'questionText':'Question'}, inplace = True)
#cleaning up the data : removing <p> & </p> 
#NEED TO CONTINUE ON FROM HERE CLEANING THIS FILE
data5['Answer'].replace("<p>",'',inplace=True, regex=True)
data5['Answer'].replace("</p>",'',inplace=True, regex=True)


#PUT ALL THE DATA TOGETHER
concate_data = pd.concat([data, data2, data3, data4])
#drops if it has any missing values
concate_data.dropna(axis=1)
concate_data.to_csv('Data/trainingData.csv')
print(concate_data.count())


############################################# GOT THIS CODE FROM SOMEWHERE  ######################################################
###############   NEED TO EDIT IT TO PREPROCESS THE DATA FROM MY CSV TRAINING FILE
#https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/
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