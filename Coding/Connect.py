#MAIN FILE WHERE THE PROGRAM IS RAN

from datetime import datetime
from glob import glob
from tkinter import *
import webbrowser

from SentimentAnalyzerGUI import SentimentApplication
from DecodeSentences import databaseUser
from ChatbotGUI import ChatApplication
from GenerateReport import createReport

root = Tk()

#Set the geometry of tkinter frame
root.geometry("450x400")


# function to open website in browser
def Website():
    webbrowser.open('file:///C:/Users/SAMSUNG/Documents/GitHub/AI-Healthcare-Pro-/Flask/templates/homepage.html')

#this holds the menu of the options available to the user e.g to talk with the chatbot, to go to the website
def menuWindow():
   
    root.resizable(width = False, height = False)
    root.config(width = 470, height = 400, bg = 'orange')

    #label is created so when clicked a new window opens
    Label(root, text= "Menu", font= "LUCIDA 15 bold").pack(pady=30)

    #Create a button to open a new window which has the chatbot
    chatbotLink = Button(root, text="Chatbot", command=ChatApplication).pack(padx=20,pady=20)
    websiteLink = Button(root, text="Website", command= Website).pack(padx=20,pady=20)
    sentimentLink = Button(root, text="Sentiment Analyzer", command=SentimentApplication).pack(padx=20,pady=20)
    createReportLink = Button(root, text="Download Report", command = createReport).pack(padx=20,pady=20)

    root.mainloop()

if __name__ == "__main__":
    day = datetime.today()
    menuWindow()
    #adds the user inputs to the database, if not here, then user input would not be commited/added to database
    databaseUser.commit()
    #drops data in table everytime the loop ends
    #databaseUser.execute("DROP TABLE User")
    #closes the database
    databaseUser.close()
