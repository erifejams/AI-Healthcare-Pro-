###TO DISPLAY THE SENTIMENT ANALYSIS TO THE USER 

import sqlite3
import pandas as pd

#creating a database
#CREATING A DATABASE FOR USER INPUT WHILE THE USER IS WRITING THEIR SENTENCES
databaseUser = sqlite3.connect('Database/UserTable.db')
cursor = databaseUser.cursor()


#read the sqlite database and display the data
dataFromDatabase = pd.read_sql("SELECT * FROM User WHERE sentence LIKE '%%' ORDER BY sentiment DESC LIMIT 1000", databaseUser)

#smooted sentiments data
#1000/2
dataFromDatabase['smootedSentiment'] = dataFromDatabase['sentiment'].rolling(int(len(dataFromDatabase)/2)).mean()
dataFromDatabase.dropna(inplace = True)
print(dataFromDatabase.tail())