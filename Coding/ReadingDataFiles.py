#READING DATA FILES
#I WANT TO PUT ALL THE DATA INTO ONE CSV FILE AS ID, QUESTION, ANSWER/ Response
import pandas as pd


#########################################################  MY CODE   ###########################################################

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
data5['Answer'].replace("<p>",'',inplace=True, regex=True)
data5['Answer'].replace("</p>",'',inplace=True, regex=True)


#PUT ALL THE DATA TOGETHER
concate_data = pd.concat([data2, data3, data, data4])
#drops if it has any missing values
concate_data.dropna(axis=1)

#REDO THE INDEX SO THAT IS STARTS FROM ONE AND GOES ON FROM THERE
concate_data.reset_index(drop=True, inplace=True)
##giving the column index as ID
concate_data.index.name = "ID"
