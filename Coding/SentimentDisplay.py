###TO DISPLAY THE SENTIMENT ANALYSIS TO THE USER 

import sqlite3
import pandas as pd

#creating a database
#CREATING A DATABASE FOR USER INPUT WHILE THE USER IS WRITING THEIR SENTENCES
databaseUser = sqlite3.connect('Database/UserTable.db')
cursor = databaseUser.cursor()


#read the sqlite database and display the data
dataFromDatabase = pd.read_sql("SELECT * FROM User WHERE sentence LIKE '%%' ORDER BY sentiment DESC LIMIT 1000", databaseUser)

def getSentiment(dataDent):
    #smooted sentiments data
    #1000/2
    dataFromDatabase['smootedSentiment'] = dataFromDatabase['sentiment'].rolling(int(len(dataFromDatabase)/2)).mean()
    dataFromDatabase.dropna(inplace = True)
    

    ''' DON'T KNOW WHY IT DIDN'T WORK IT GAVE PUT ALL THE NUMBERS INTO ONE THING E.G ALL NEGATIVE
    for i in dataFromDatabase['smootedSentiment']:
        if i > 0.1:
            dataFromDatabase['sentimentEvaluation'] = "pos"
        elif i < 0: 
            dataFromDatabase['sentimentEvaluation'] = "neg"
        elif i == 0: 
            dataFromDatabase['sentimentEvaluation'] = "neutral"
    '''
    dataDent = dataFromDatabase['smootedSentiment']
    return dataDent 


#getSentiment()
