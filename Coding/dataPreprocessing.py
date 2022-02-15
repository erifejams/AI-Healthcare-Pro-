#I WANT TO PUT ALL THE DATA INTO ONE CSV FILE AS ID, QUESTION, ANSWER
#DO DATA PREPROCESSING STEPS WHICH ARE: 
    #Removing punctuations like . , ! $( ) * % @
    #Removing URLs
    #Removing Stop words
    #Lower casing
    #Tokenization
    #Stemming
    #Lemmatization

import pandas as pd

#reading the data
data = pd.read_csv("Psych_data.csv")
data.head()