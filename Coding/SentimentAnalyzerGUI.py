###THIS IS TO CREATE A GRAPHICAL INTERFACE FOR THE SENTIMENT ANAYLZER


from tkinter import *
from SentimentDisplay import getSentiment

class SentimentApplication:

    def __init__(self):
        self.root = Tk() #gonna be used as a parent as reference in the rest of the code to connect with tk(in tinker)
        self._setup_main_window()

    ###to run the application
    def run(self):
        self.root.mainloop()


    ##this is how the window will look(apperance wise) and contains features such as scrolling
    def _setup_main_window(self):

        #title of the window that appears
        self.root.title('Sentiment Analyzer')
        self.root.resizable(width = False, height = False)
        self.root.config(width = 570, height = 550, bg = 'black')

        #the place at the top of the chat where it says what is inserted in the text
        head_label = Label(self.root, bg = 'light yellow', fg = 'black', text = "INTA CONVERSATION ANALYSIS", pady = 10) #pady to move down
        head_label.place(relwidth = 1)

        #this is the area where the sentiment and text is going to appear
        #fg is the color of the text
        self.sentimentWindow = Text(self.root, bd = 1, bg = 'white', width = 400, height = 500, fg = 'black', padx = 0, pady = 0)
        self.sentimentWindow.place(relheight = 1, relwidth = 1, rely = 0.08) #the entire width of the page is used
        #so the users cursor won't be able to click on the box
        self.sentimentWindow.configure(cursor = "arrow", state = DISABLED)


    def getInput(self, sentiment):
        #self.sentimentWindow.delete(0, END)
        #gets the chatbot to talk
        sentimentResponse = f"{getSentiment(sentiment)}\n"
        ###message is inserted into the chat box
        self.sentimentWindow.configure(state = NORMAL)
        self.sentimentWindow.insert(END, sentimentResponse)
        #disable it again
        self.sentimentWindow.configure(state = DISABLED)
        #Lets you see the last message that is in chat box
        self.sentimentWindow.see(END)
        

