import pandas as pd
import yaml
import json
import nltk


with open(r'Data/Original/personality/ai.yml') as file:
    documents = yaml.full_load(file)
    QuestionConv = []
    rightFormat = []
    i = 1
 
    #print(item, ":", doc)
    #turns the dictionary to a string
    QuestionConv = json.dumps(documents)
    #print(QuestionConv)
    #splits the sentence by looking for ],
    QuestionConv = QuestionConv.split("], ")

    #this splits the data, so that i can get it has individual strings
    while(i != len(QuestionConv)):
        rightFormat.append(QuestionConv[i])
        i+=1
    
     
    #removes the [  and replaces it with an empty space
    rightFormat = [k.replace("[", "") for k in rightFormat]
    rightFormat = [k.replace('"', "") for k in rightFormat]
    #removing the conversations from the start
    rightFormat = [k.replace('conversations: ', "") for k in rightFormat]

    # define punctuation
    punctuations = ''','''
    rightFormat = [k.replace(punctuations, " BREAKHERE ") for k in rightFormat]

    
    print(rightFormat)
    
    #dataNow = pd.DataFrame(rightFormat , columns = ['Question'])
    #print(dataNow)
    #dataNow= rightFormat.apply(lambda x:remove_punctuation_tokenize(str(x)))






  