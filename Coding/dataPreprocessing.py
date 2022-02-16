#I WANT TO PUT ALL THE DATA INTO ONE CSV FILE AS ID, QUESTION, ANSWER(response)
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
#https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/
data = pd.read_csv('Data/Psych_data.csv', encoding="ISO-8859-1")
#putting only the column that I needed from Psych_data.csv
data = data[['Answer', 'Question']]
#this is to read the data to a csv file
#data.to_csv('Data/trainingData.csv')


#turn txt file to a csv file
#then put it under a column
#then find the word e.g human 1 and put it under a 2 separate columns e.g Question and answer

humanConv = list()
firstConv = list()
with open('Data/human_chat.txt', encoding="utf8") as file:
   # NormalConv = file.readlines()
    for line in file:
        humanConv.append(line.strip())
        if humanConv.line.str.contains(r'\bHuman 1\b'):
            firstConv.append(line)


print(humanConv)
"""
    for i in range(len(NormalConv)):
        if '?' in NormalConv[i]:
            NormalConv['Question'] = NormalConv[i]
        NormalConv['Answer'] = NormalConv[i]
"""
   
#print(NormalConv)