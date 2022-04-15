from tkinter import *
import webbrowser

from SentimentAnalyzerGUI import SentimentApplication
from DecodeSentences import databaseUser
from ChatbotGUI import ChatApplication

root = Tk()

#Set the geometry of tkinter frame
root.geometry("750x250")


# function to open linkedin in browser
def Website():
    webbrowser.open("https://github.com/erifejams/AI-Healthcare-Pro-/blob/main/Website/homepage.html")

#this holds the menu of the options available to the user e.g to talk with the chatbot, to go to the website
def menuWindow():
   
    root.resizable(width = False, height = False)
    root.config(width = 470, height = 350, bg = 'orange')

    #label is created so when clicked a new window opens
    Label(root, text= "Menu", font= "LUCIDA 15 bold").pack(pady=30)

    #Create a button to open a new window which has the chatbot
    chatbotLink = Button(root, text="Chatbot", command=ChatApplication).pack(padx=20,pady=20)
    websiteLink = Button(root, text="Website", command=Website).pack(padx=20,pady=20)

    root.mainloop()

if __name__ == "__main__":
    menuWindow()
    sentimentApp = SentimentApplication()
    sentimentApp.run()

    #adds the user inputs to the database, if not here, then user input would not be commited/added to database
    databaseUser.commit()
    #closes the database
    databaseUser.close()