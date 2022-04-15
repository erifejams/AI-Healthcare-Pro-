#THIS IS THE MAIN FILE WHERE IT RUNS


#from nltk
from SentimentAnalyzerGUI import SentimentApplication
from DecodeSentences import databaseUser
from ChatbotGUI import ChatApplication
from Flask.templates.app import *
#main
if __name__ == "__main__":
    #gets the interface to show up and the chatbot to talk
    appli.run(debug= True)
    app = ChatApplication()
    sentimentApp = SentimentApplication()
    app.run()
    sentimentApp.run()

    #adds the user inputs to the database, if not here, then user input would not be commited/added to database
    databaseUser.commit()
    #closes the database
    databaseUser.close()
        
